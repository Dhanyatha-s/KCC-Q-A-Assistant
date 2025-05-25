import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from typing import List, Dict, Any, Optional
import warnings
import os
warnings.filterwarnings("ignore")

class OfflineHuggingFaceLLM:
    """
    Offline HuggingFace LLM wrapper that works without internet after initial setup.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small", cache_dir: str = "./models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.is_t5_model = "t5" in model_name.lower()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        self.load_model()
    
    def load_model(self):
        """Load HuggingFace model with offline capability."""
        print(f"Loading HuggingFace model: {self.model_name}")
        
        # Configure for CPU/GPU efficiency
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            # Load with cache directory for offline access
            if self.is_t5_model:
                print("Loading T5 model...")
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir,
                    local_files_only=False  # Set to True after first download
                )
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False,  # Set to True after first download
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                ).to(device)
                
                # Create pipeline
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            else:
                # GPT-style models
                print("Loading GPT-style model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    pad_token_id=self.tokenizer.eos_token_id
                ).to(device)
                
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            print(f"✅ Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {str(e)}")
            print("Falling back to distilgpt2...")
            self._load_fallback_model(device)
    
    def _load_fallback_model(self, device):
        """Load a simple fallback model."""
        try:
            self.model_name = "distilgpt2"
            self.is_t5_model = False
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilgpt2",
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                cache_dir=self.cache_dir
            ).to(device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                max_length=256,
                do_sample=True,
                temperature=0.8
            )
            print("✅ Fallback model loaded successfully")
            
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {str(fallback_error)}")
            self.pipeline = None
    
    def generate(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using HuggingFace pipeline."""
        if self.pipeline is None:
            return "Model not loaded properly. Please check the setup."
        
        try:
            # Clean and prepare prompt
            prompt = prompt.strip()
            if not prompt:
                return "Please provide a valid question."
            
            if self.is_t5_model:
                # T5 models work with instructions
                response = self.pipeline(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                generated_text = response[0]['generated_text'].strip()
            else:
                # GPT-style models
                response = self.pipeline(
                    prompt,
                    max_new_tokens=min(max_length, 150),  # Limit tokens to avoid memory issues
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    truncation=True
                )
                generated_text = response[0]['generated_text']
                
                # Remove the original prompt from response
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            return generated_text if generated_text else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error while generating a response. Please try again with a simpler question."
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the generated response."""
        if not text:
            return ""
        
        # Remove common artifacts
        text = text.replace("<|endoftext|>", "")
        text = text.replace("[PAD]", "")
        text = text.replace("[UNK]", "")
        
        # Split by sentences and take the first few complete ones
        sentences = text.split('.')
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Avoid very short fragments
                clean_sentences.append(sentence)
            if len(clean_sentences) >= 3:  # Limit to 3 sentences
                break
        
        if clean_sentences:
            result = '. '.join(clean_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        
        return text.strip()

class OfflineKCCQueryHandler:
    """
    Offline KCC Query Handler that works without internet connectivity.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small", cache_dir: str = "./models"):
        """
        Initialize with offline-capable models.
        
        Recommended offline models:
        1. "google/flan-t5-small" - Good balance, smaller size
        2. "distilgpt2" - Very lightweight, fast
        3. "microsoft/DialoGPT-small" - Good for conversations
        """
        print(f"🚀 Initializing Offline KCC Query Handler with model: {model_name}")
        self.llm = OfflineHuggingFaceLLM(model_name=model_name, cache_dir=cache_dir)
        self.setup_prompts()
        print("✅ Handler initialized successfully")
    
    def setup_prompts(self):
        """Setup prompt templates optimized for agricultural Q&A."""
        
        # Template for when context is found
        self.context_template = """You are an agricultural expert. Answer the farming question using the KCC database information provided.

KCC Information:
{context}

Question: {query}

Answer:"""
        
        # Template for when no context is found
        self.no_context_template = """You are an agricultural expert. Answer this farming question with general guidance:

Question: {query}

Answer:"""
        
        # Agricultural knowledge base for fallback
        self.fallback_responses = {
            'pest': "For pest control, use integrated pest management (IPM) including biological controls, neem-based pesticides, and crop rotation. Monitor regularly and treat early.",
            'fertilizer': "Use balanced NPK fertilizers based on soil test results. Apply organic compost and follow recommended schedules for your crop and region.",
            'disease': "Prevent diseases through proper spacing, good drainage, disease-resistant varieties, and fungicide application when necessary.",
            'irrigation': "Practice efficient water management with drip irrigation, mulching, and watering during cooler hours to reduce evaporation.",
            'crop': "Choose varieties suited to your climate and soil. Follow proper planting schedules and crop rotation practices."
        }
    
    def generate_response_with_context(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response when relevant context is found."""
        if not context_chunks:
            return self.generate_response_without_context(query)
        
        try:
            # Combine context from top chunks
            context_text = ""
            sources = []
            
            for i, chunk in enumerate(context_chunks[:2]):  # Use top 2 chunks to avoid token limits
                context_text += f"{chunk.get('response', chunk.get('text', ''))}\n"
                sources.append({
                    'state': chunk.get('metadata', {}).get('state', 'Unknown'),
                    'crop': chunk.get('metadata', {}).get('crop', 'Unknown'),
                    'category': chunk.get('metadata', {}).get('category', 'General'),
                    'similarity': chunk.get('similarity_score', 0.0)
                })
            
            # Create concise prompt
            prompt = f"""Answer this farming question using the provided information:

Information: {context_text.strip()[:300]}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.llm.generate(prompt, max_length=150)
            
            if response and len(response.strip()) > 10:
                return {
                    'answer': response,
                    'source_type': 'KCC Database',
                    'sources': sources,
                    'confidence': max([chunk.get('similarity_score', 0.0) for chunk in context_chunks]),
                    'num_sources': len(context_chunks)
                }
            else:
                return self.generate_fallback_response(query)
            
        except Exception as e:
            print(f"Error generating contextual response: {str(e)}")
            return self.generate_fallback_response(query)
    
    def generate_response_without_context(self, query: str) -> Dict[str, Any]:
        """Generate response when no relevant context is found."""
        try:
            prompt = f"""Answer this farming question with practical advice:

Question: {query}

Answer:"""
            
            response = self.llm.generate(prompt, max_length=120)
            
            if response and len(response.strip()) > 10:
                return {
                    'answer': response,
                    'source_type': 'General Knowledge',
                    'sources': [],
                    'confidence': 0.5,
                    'num_sources': 0,
                    'suggestion': 'For specific regional advice, contact your local agricultural extension office or KCC helpline at 1800-180-1551.'
                }
            else:
                return self.generate_fallback_response(query)
            
        except Exception as e:
            print(f"Error generating no-context response: {str(e)}")
            return self.generate_fallback_response(query)
    
    def generate_fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate rule-based fallback response when LLM fails."""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        response = "I understand you're asking about farming. "
        
        if any(word in query_lower for word in ['pest', 'insect', 'bug', 'control']):
            response += self.fallback_responses['pest']
        elif any(word in query_lower for word in ['fertilizer', 'nutrient', 'npk', 'manure']):
            response += self.fallback_responses['fertilizer']
        elif any(word in query_lower for word in ['disease', 'fungus', 'infection', 'rot']):
            response += self.fallback_responses['disease']
        elif any(word in query_lower for word in ['water', 'irrigation', 'drought']):
            response += self.fallback_responses['irrigation']
        else:
            response += self.fallback_responses['crop']
        
        response += " For detailed guidance specific to your region and crop, please contact KCC at 1800-180-1551."
        
        return {
            'answer': response,
            'source_type': 'Fallback',
            'sources': [],
            'confidence': 0.3,
            'num_sources': 0
        }

def test_offline_handler():
    """Test the offline handler functionality."""
    print("🧪 Testing Offline KCC Handler...")
    
    handler = OfflineKCCQueryHandler(model_name="google/flan-t5-small")
    
    test_queries = [
        "How to control pests in tomato crop?",
        "What fertilizer is best for wheat?",
        "How to prevent fungal diseases in rice?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        response = handler.generate_response_without_context(query)
        print(f"Answer: {response['answer']}")
        print(f"Source: {response['source_type']}")

if __name__ == "__main__":
    test_offline_handler()