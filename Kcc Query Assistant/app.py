import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Set environment variables for better PyTorch compatibility
os.environ['TORCH_SERIALIZATION_CHECK'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_JIT_USE_NVF'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'



try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Charts will be disabled.")

# Import custom modules with better error handling
try:
    from vector_store import VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

try:
    from offline_llm import OfflineKCCQueryHandler
    OFFLINE_LLM_AVAILABLE = True
except ImportError:
    OFFLINE_LLM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="KCC Agricultural Q&A Assistant (Offline)",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KCCDataLoader:
    """
    Enhanced KCC data loader to properly handle kcc_processed_data.csv
    """
    
    def __init__(self, csv_path: str = "kcc_processed_data.csv"):
        self.csv_path = csv_path
        self.data = None
        self.loaded = False
        self.queries = []
        self.answers = []
        
    def load_kcc_data(self):
        """Load KCC data from CSV file with robust error handling"""
        try:
            if not os.path.exists(self.csv_path):
                st.error(f"❌ KCC data file not found: {self.csv_path}")
                return False
            
            # Load CSV with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.csv_path, encoding=encoding)
                    st.success(f"✅ Successfully loaded KCC data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.data is None:
                st.error("❌ Failed to load CSV with any encoding")
                return False
            
            # Display basic info about the data
            st.info(f"📊 Loaded {len(self.data)} records from KCC database")
            st.write("**Column names:**", list(self.data.columns))
            
            # Check for required columns
            required_columns = self.detect_columns()
            if not required_columns:
                return False
            
            # Process and clean the data
            self.process_data()
            self.loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading KCC data: {str(e)}")
            return False
    
    def detect_columns(self):
        """Detect and map column names (case-insensitive)"""
        columns = self.data.columns
        column_mapping = {}
        
        # Possible column names for queries/questions
        query_columns = ['query', 'question', 'kccquery', 'kcc_query', 'farmer_query', 'q']
        answer_columns = ['answer', 'kccans', 'kcc_ans', 'kcc_answer', 'response', 'a']
        
        # Find query column
        query_col = None
        for col in columns:
            if col.lower().strip() in [q.lower() for q in query_columns]:
                query_col = col
                break
        
        # Find answer column  
        answer_col = None
        for col in columns:
            if col.lower().strip() in [a.lower() for a in answer_columns]:
                answer_col = col
                break
        
        if not query_col:
            st.error("❌ Could not find query/question column in CSV")
            st.write("Available columns:", list(columns))
            return False
            
        if not answer_col:
            st.error("❌ Could not find answer/KccAns column in CSV")
            st.write("Available columns:", list(columns))
            return False
        
        self.query_column = query_col
        self.answer_column = answer_col
        
        st.success(f"✅ Detected columns - Query: '{query_col}', Answer: '{answer_col}'")
        return True
    
    def process_data(self):
        """Process and clean the loaded data"""
        try:
            # Remove rows with missing values in key columns
            initial_count = len(self.data)
            self.data = self.data.dropna(subset=[self.query_column, self.answer_column])
            
            if len(self.data) < initial_count:
                st.warning(f"⚠️ Removed {initial_count - len(self.data)} rows with missing data")
            
            # Clean text data
            self.data[self.query_column] = self.data[self.query_column].astype(str).str.strip()
            self.data[self.answer_column] = self.data[self.answer_column].astype(str).str.strip()
            
            # Remove empty strings
            self.data = self.data[
                (self.data[self.query_column] != '') & 
                (self.data[self.answer_column] != '')
            ]
            
            # Extract queries and answers as lists for easy access
            self.queries = self.data[self.query_column].tolist()
            self.answers = self.data[self.answer_column].tolist()
            
            st.success(f"✅ Processed {len(self.data)} valid Q&A pairs")
            
            # Show sample data
            if len(self.data) > 0:
                st.subheader("📋 Sample KCC Data")
                sample_size = min(3, len(self.data))
                for i in range(sample_size):
                    with st.expander(f"Sample {i+1}: {self.queries[i][:100]}..."):
                        st.write(f"**Q:** {self.queries[i]}")
                        st.write(f"**A:** {self.answers[i]}")
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            raise

class EnhancedFallbackQueryHandler:
    """
    Enhanced fallback handler that uses KCC CSV data
    """
    
    def __init__(self, csv_path: str = "kcc_processed_data.csv"):
        self.kcc_loader = KCCDataLoader(csv_path)
        self.kcc_loaded = False
        
        # Initialize basic agricultural knowledge as backup
        self.basic_knowledge = {
            'pest_control': {
                'keywords': ['pest', 'insect', 'bug', 'aphid', 'whitefly', 'bollworm', 'caterpillar'],
                'response': "For pest control, use Integrated Pest Management (IPM): monitor regularly, use beneficial insects, apply neem-based pesticides, practice crop rotation."
            },
            'fertilizer': {
                'keywords': ['fertilizer', 'nutrient', 'npk', 'manure', 'compost'],
                'response': "For fertilizer: Get soil test first, use balanced NPK, apply organic compost, split nitrogen application."
            },
            'disease': {
                'keywords': ['disease', 'fungus', 'blight', 'wilt', 'rot'],
                'response': "Disease prevention: Use resistant varieties, maintain spacing, ensure drainage, remove infected debris."
            },
            'irrigation': {
                'keywords': ['water', 'irrigation', 'drought', 'watering'],
                'response': "Efficient irrigation: Use drip system, water early morning, apply mulch, monitor soil moisture."
            }
        }
    
    def load_kcc_data(self):
        """Load KCC data"""
        try:
            self.kcc_loaded = self.kcc_loader.load_kcc_data()
            return self.kcc_loaded
        except Exception as e:
            st.error(f"Failed to load KCC data: {str(e)}")
            return False
    
    def search_kcc_data(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search KCC data for similar queries"""
        if not self.kcc_loaded:
            return []
        
        query_lower = query.lower()
        results = []
        
        try:
            for i, (q, a) in enumerate(zip(self.kcc_loader.queries, self.kcc_loader.answers)):
                q_lower = q.lower()
                
                # Simple keyword matching
                query_words = set(query_lower.split())
                q_words = set(q_lower.split())
                
                # Calculate similarity score
                common_words = query_words.intersection(q_words)
                if common_words:
                    similarity = len(common_words) / len(query_words.union(q_words))
                    
                    results.append({
                        'query': q,
                        'answer': a,
                        'similarity': similarity,
                        'index': i
                    })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"Error searching KCC data: {str(e)}")
            return []
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response using KCC data first, then fallback"""
        try:
            # First try to find answer in KCC data
            if self.kcc_loaded:
                kcc_results = self.search_kcc_data(query, top_k=3)
                
                if kcc_results and kcc_results[0]['similarity'] > 0.3:
                    best_match = kcc_results[0]
                    return {
                        'answer': best_match['answer'],
                        'source_type': 'KCC Database',
                        'sources': [best_match['query']],
                        'confidence': min(0.9, best_match['similarity'] + 0.2),
                        'num_sources': 1,
                        'suggestion': 'This answer is from the official KCC database.'
                    }
            
            # Fallback to basic knowledge
            query_lower = query.lower()
            for category, data in self.basic_knowledge.items():
                if any(keyword in query_lower for keyword in data['keywords']):
                    return {
                        'answer': data['response'],
                        'source_type': 'Basic Agricultural Knowledge',
                        'sources': [],
                        'confidence': 0.6,
                        'num_sources': 0,
                        'suggestion': 'For specific advice, contact KCC helpline at 1800-180-1551.'
                    }
            
            # Generic response if no match found
            return {
                'answer': """I understand you're asking about farming. Here are general guidelines:
                
1. Always start with soil testing and choose appropriate varieties
2. Use integrated pest management (IPM) for pest control  
3. Practice crop rotation and maintain soil health
4. Use efficient irrigation and monitor water needs
5. Follow recommended fertilizer schedules

For specific regional advice, contact your local agricultural extension office or KCC helpline at 1800-180-1551.""",
                'source_type': 'General Agricultural Guidance',
                'sources': [],
                'confidence': 0.4,
                'num_sources': 0,
                'suggestion': 'Contact KCC helpline for specific regional advice.'
            }
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {
                'answer': 'I encountered an error. Please contact KCC helpline at 1800-180-1551.',
                'source_type': 'Error',
                'sources': [],
                'confidence': 0.0,
                'num_sources': 0,
                'suggestion': 'Try rephrasing your question or contact KCC directly.'
            }

class FallbackQueryHandler:
    """
    Simple rule-based fallback query handler when LLM is not available.
    """
    
    def __init__(self):
        self.knowledge_base = {
            'pest_control': {
                'keywords': ['pest', 'insect', 'bug', 'aphid', 'whitefly', 'bollworm', 'caterpillar'],
                'responses': [
                    "For pest control, use Integrated Pest Management (IPM) approach: 1) Monitor crops regularly, 2) Use beneficial insects like ladybugs, 3) Apply neem-based organic pesticides, 4) Practice crop rotation, 5) Remove infected plant parts immediately.",
                    "Common organic pest control methods: Neem oil spray, soap solution, marigold companion planting, sticky traps, and beneficial predatory insects. Always identify the pest first for targeted treatment.",
                    "For severe infestations, consult your local agricultural extension officer. Use chemical pesticides only as last resort and follow all safety guidelines."
                ]
            },
            'fertilizer': {
                'keywords': ['fertilizer', 'nutrient', 'npk', 'manure', 'compost', 'nitrogen', 'phosphorus', 'potassium'],
                'responses': [
                    "Balanced fertilizer application: Get soil test done first. Generally use NPK 19:19:19 for most crops. Apply organic compost (5-10 tons/hectare) before planting. Split nitrogen application in 2-3 doses.",
                    "Organic fertilizer options: Farmyard manure, vermicompost, green manure crops (like dhaincha), bone meal, and biofertilizers. These improve soil health long-term.",
                    "Timing is crucial: Apply phosphorus and potassium at planting. Split nitrogen - 1/3 at planting, 1/3 at vegetative stage, 1/3 at flowering/grain filling stage."
                ]
            },
            'disease': {
                'keywords': ['disease', 'fungus', 'blight', 'wilt', 'rot', 'leaf spot', 'rust'],
                'responses': [
                    "Disease prevention is key: Use disease-resistant varieties, maintain proper plant spacing, ensure good drainage, avoid overhead irrigation, and remove infected plant debris.",
                    "Common fungal diseases can be controlled with copper-based fungicides, proper air circulation, and avoiding water on leaves. Apply preventive sprays during humid weather.",
                    "For viral diseases: Control vector insects, use virus-free seeds, maintain field hygiene, and remove infected plants immediately to prevent spread."
                ]
            },
            'irrigation': {
                'keywords': ['water', 'irrigation', 'drought', 'watering', 'drip', 'sprinkler'],
                'responses': [
                    "Efficient irrigation: Use drip irrigation for 40-60% water savings. Water early morning or evening. Apply mulch to reduce evaporation. Check soil moisture at root zone depth.",
                    "Drought management: Choose drought-tolerant varieties, practice rainwater harvesting, use mulching, reduce plant population if needed, and apply anti-transpirants.",
                    "Overwatering causes root rot and nutrient leaching. Monitor soil moisture with finger test or moisture meter. Most crops need 1-2 inches water per week."
                ]
            },
            'soil': {
                'keywords': ['soil', 'ph', 'fertility', 'organic matter', 'erosion'],
                'responses': [
                    "Soil health improvement: Add organic matter regularly, practice crop rotation, use cover crops, minimize tillage, and maintain soil pH between 6.0-7.5 for most crops.",
                    "Soil testing should be done every 2-3 years. Test for pH, NPK, organic matter, and micronutrients. Adjust based on recommendations for your specific crop.",
                    "Prevent soil erosion with contour farming, terracing, cover crops, windbreaks, and maintaining vegetation buffers around fields."
                ]
            }
        }
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response based on keyword matching."""
        query_lower = query.lower()
        best_match = None
        best_score = 0
        
        # Find best matching category
        for category, data in self.knowledge_base.items():
            score = sum(1 for keyword in data['keywords'] if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_match = category
        
        if best_match and best_score > 0:
            import random
            response = random.choice(self.knowledge_base[best_match]['responses'])
            confidence = min(0.8, 0.4 + (best_score * 0.1))
        else:
            response = """I understand you're asking about farming. Here are some general guidelines:
            
1. Always start with soil testing and choose appropriate varieties for your region
2. Use integrated pest management (IPM) for pest control
3. Practice crop rotation and maintain soil health with organic matter
4. Use efficient irrigation methods and monitor water needs
5. Follow recommended fertilizer schedules based on soil test results

For specific advice, please contact your local agricultural extension office or KCC helpline at 1800-180-1551."""
            confidence = 0.3
        
        return {
            'answer': response,
            'source_type': 'Fallback Knowledge Base',
            'sources': [],
            'confidence': confidence,
            'num_sources': 0,
            'suggestion': 'For specific regional advice, contact your local agricultural extension office.'
        }

class SimpleVectorStore:
    """
    Simple fallback vector store using basic text matching when FAISS is not available.
    """
    
    def __init__(self):
        self.chunks = []
        self.loaded = False
    
    def load_vector_store(self):
        """Try to load chunks from JSON file."""
        try:
            chunks_path = os.path.join("vector_store", "chunks.json")
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                self.loaded = True
                return True
        except Exception as e:
            st.warning(f"Could not load vector store: {e}")
        return False
    
    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Simple keyword-based search."""
        if not self.loaded:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for chunk in self.chunks:
            text = chunk.get('text', '') + ' ' + chunk.get('query', '')
            text_words = set(text.lower().split())
            
            # Calculate simple word overlap score
            common_words = query_words.intersection(text_words)
            if common_words:
                score = len(common_words) / len(query_words.union(text_words))
                if score >= threshold:
                    chunk_copy = chunk.copy()
                    chunk_copy['similarity_score'] = score
                    results.append(chunk_copy)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

class OfflineKCCStreamlitApp:
    def __init__(self):
        """Initialize the offline Streamlit app with fallback handling."""
        self.vector_store = None
        self.query_handler = None
        self.fallback_handler = FallbackQueryHandler()
        self.kcc_handler = EnhancedFallbackQueryHandler("kcc_processed_data.csv")
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'vector_store_loaded' not in st.session_state:
            st.session_state.vector_store_loaded = False
        if 'llm_loaded' not in st.session_state:
            st.session_state.llm_loaded = False
        if 'kcc_csv_loaded' not in st.session_state:
            st.session_state.kcc_csv_loaded = False
        if 'total_queries' not in st.session_state:
            st.session_state.total_queries = 0
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        if 'response_times' not in st.session_state:
            st.session_state.response_times = []
        if 'confidence_scores' not in st.session_state:
            st.session_state.confidence_scores = []
        if 'use_fallback' not in st.session_state:
            st.session_state.use_fallback = not OFFLINE_LLM_AVAILABLE
    
    def show_offline_notice(self):
        """Display offline capability notice."""
        st.markdown("""
        <div style="background-color: #191832; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4; margin-bottom: 20px;">
            <strong>🔒 Offline Mode:</strong> This application can work completely offline. 
            If advanced AI models fail to load, it will use KCC CSV data and built-in agricultural knowledge base.
        </div>
        """, unsafe_allow_html=True)
    
    def load_kcc_csv_data(self):
        """Load KCC CSV data"""
        with st.spinner("Loading KCC database..."):
            success = self.kcc_handler.load_kcc_data()
            if success:
                st.session_state.kcc_csv_loaded = True
                st.success("✅ KCC database loaded successfully!")
                return True
            else:
                st.warning("⚠️ Could not load KCC database. Using basic knowledge.")
                return False
    
    def load_vector_store(self, vector_store_dir: str = "vector_store"):
        """Load vector store with fallback handling."""
        try:
            if VECTOR_STORE_AVAILABLE:
                if not st.session_state.vector_store_loaded:
                    with st.spinner("Loading vector store..."):
                        if not os.path.exists(vector_store_dir):
                            st.warning(f"Vector store directory '{vector_store_dir}' not found. Using simple search.")
                            self.vector_store = SimpleVectorStore()
                            return self.vector_store.load_vector_store()
                        
                        self.vector_store = VectorStore(vector_store_dir)
                        self.vector_store.load_vector_store()
                        st.session_state.vector_store_loaded = True
                    st.success("✅ Vector store loaded successfully!")
                return True
            else:
                # Use simple fallback vector store
                self.vector_store = SimpleVectorStore()
                success = self.vector_store.load_vector_store()
                if success:
                    st.session_state.vector_store_loaded = True
                    st.info("ℹ️ Using simple text search (FAISS not available)")
                return success
                
        except Exception as e:
            st.warning(f"Vector store loading failed: {str(e)}. Continuing without vector search.")
            return False
    
    def load_llm_handler(self, model_name: str = "google/flan-t5-small", cache_dir: str = "./models"):
        """Load LLM handler with comprehensive fallback."""
        try:
            if OFFLINE_LLM_AVAILABLE:
                if not st.session_state.llm_loaded:
                    with st.spinner(f"Loading AI model: {model_name}..."):
                        # Force CPU-only loading to avoid PyTorch CUDA issues
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass  # PyTorch not available, that's fine
                        
                        self.query_handler = OfflineKCCQueryHandler(
                            model_name=model_name, 
                            cache_dir=cache_dir,
                            device='cpu'  # Force CPU to avoid GPU issues
                        )
                        st.session_state.llm_loaded = True
                        st.session_state.use_fallback = False
                    st.success("✅ AI model loaded successfully!")
                return True
            else:
                st.info("ℹ️ Advanced AI models not available. Using KCC database and built-in knowledge.")
                st.session_state.use_fallback = True
                return True
                
        except Exception as e:
            st.warning(f"AI model loading failed: {str(e)}. Using fallback knowledge base.")
            st.session_state.use_fallback = True
            return True

    def process_query(self, query: str, similarity_threshold: float = 0.3, top_k: int = 5) -> Dict[str, Any]:
        """Process query with comprehensive fallback handling."""
        start_time = time.time()
        
        try:
            # Priority order: KCC CSV -> AI Model -> Basic Fallback
            
            # First priority: KCC CSV data
            if st.session_state.kcc_csv_loaded and self.kcc_handler.kcc_loaded:
                response = self.kcc_handler.generate_response(query)
                if response['confidence'] > 0.3:  # Good match found in KCC data
                    response['timestamp'] = datetime.now()
                    response['query'] = query
                    response['error'] = False
                    response['response_time'] = time.time() - start_time
                    return response
            
            # Second priority: AI Model with vector search
            should_use_ai = (
                not st.session_state.get('use_fallback', True) and
                OFFLINE_LLM_AVAILABLE and
                self.query_handler is not None
            )
            
            if should_use_ai:
                # Search for similar content from vector store
                similar_chunks = []
                if self.vector_store:
                    try:
                        similar_chunks = self.vector_store.search_similar(
                            query, 
                            top_k=top_k, 
                            threshold=similarity_threshold
                        )
                    except Exception as vs_error:
                        st.warning(f"Vector search failed: {str(vs_error)}")
                
                # Generate AI-enhanced response
                try:
                    if similar_chunks:
                        response = self.query_handler.generate_response_with_context(query, similar_chunks)
                    else:
                        response = self.query_handler.generate_response_without_context(query)
                except Exception as llm_error:
                    st.warning(f"AI model failed: {str(llm_error)}. Using fallback.")
                    response = self.fallback_handler.generate_response(query)
            else:
                # Third priority: Basic fallback knowledge
                response = self.fallback_handler.generate_response(query)
            
            # Add metadata
            response_time = time.time() - start_time
            response.update({
                'timestamp': datetime.now(),
                'query': query,
                'error': False,
                'response_time': response_time
            })
            
            # Update session state statistics
            st.session_state.response_times.append(response_time)
            st.session_state.confidence_scores.append(response.get('confidence', 0.0))
            
            return response
            
        except Exception as e:
            st.error(f"Query processing error: {str(e)}")
            return {
                'answer': 'I encountered an error. Please try again or contact KCC helpline at 1800-180-1551.',
                'source_type': 'Error',
                'sources': [],
                'confidence': 0.0,
                'error': True,
                'timestamp': datetime.now(),
                'query': query,
                'response_time': time.time() - start_time
            }

    def display_response(self, response: Dict[str, Any]):
        """Display response with improved formatting."""
        if response['error']:
            st.markdown(f"""
            <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336;width: 90vw; 
                max-width: 1200px;">
                <h4>❌ Error</h4>
                <p>{response['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Determine styling based on source type
        if response['source_type'] == 'KCC Database':
            bg_color = "#e8f5e8"
            border_color = "#4caf50"
            icon = "🎯"
        elif 'AI' in response['source_type'] or response['source_type'] == 'General Knowledge':
            bg_color = "#e3f2fd"
            border_color = "#2196f3"
            icon = "🤖"
        else:
            bg_color = "#190f0039"
            border_color = "#0099ff"
            icon = "📚"
        
        # Display answer
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid {border_color}; margin: 10px auto; width: 80vw; 
                max-width: 1000px;">
            <h4>{icon} Assistant Response</h4>
            <p style="font-size: 16px; line-height: 1.6;">{response['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

        

        # col1, col2, col3, col4 = st.columns(4)

        # with col1:
        #     confidence = response.get('confidence', 0.0)
        #     if confidence > 0.7:
        #         conf_color = "#4caf50"
        #     elif confidence > 0.4:
        #         conf_color = "#ff9800"
        #     else:
        #         conf_color = "#f44336"
        #     st.markdown(f"**Confidence:** <span style='color: {conf_color}; font-weight: bold;'>{confidence:.1%}</span>", 
        #                 unsafe_allow_html=True)

        # with col2:
        #     st.markdown(f"**Source:** {response['source_type']}")

        # with col3:
        #     st.markdown(f"**References:** {response.get('num_sources', 0)}")

        # with col4:
        #     response_time = response.get('response_time', 0.0)
        #     st.markdown(f"**Time:** {response_time:.2f}s")

        # st.markdown("</div>", unsafe_allow_html=True)  # Close the custom container

        # # Display suggestion
        # if 'suggestion' in response and response['suggestion']:
        #     st.info(f"💡 **Suggestion:** {response['suggestion']}")

        # # Display sources if available
        # if response.get('sources') and len(response['sources']) > 0:
        #     with st.expander("📋 View Sources"):
        #         for i, source in enumerate(response['sources'], 1):
        #             st.write(f"{i}. {source}")

        
        # # Display metadata
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     confidence = response.get('confidence', 0.0)
        #     if confidence > 0.7:
        #         conf_color = "#4caf50"
        #     elif confidence > 0.4:
        #         conf_color = "#ff9800"
        #     else:
        #         conf_color = "#f44336"
        #     st.markdown(f"**Confidence:** <span style='color: {conf_color}; font-weight: bold;'>{confidence:.1%}</span>", 
        #                unsafe_allow_html=True)
        
        # with col2:
        #     st.markdown(f"**Source:** {response['source_type']}")
        
        # with col3:
        #     st.markdown(f"**References:** {response.get('num_sources', 0)}")
        
        # with col4:
        #     response_time = response.get('response_time', 0.0)
        #     st.markdown(f"**Time:** {response_time:.2f}s")
        
        # # Display suggestion
        # # Display suggestion
        # if 'suggestion' in response and response['suggestion']:
        #     st.info(f"💡 **Suggestion:** {response['suggestion']}")
        
        # # Display sources if available
        # if response.get('sources') and len(response['sources']) > 0:
        #     with st.expander("📋 View Sources"):
        #         for i, source in enumerate(response['sources'], 1):
        #             st.write(f"{i}. {source}")

    def show_statistics(self):
        """Display application statistics."""
        if st.session_state.total_queries > 0:
            st.subheader("📊 Usage Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", st.session_state.total_queries)
            
            with col2:
                if st.session_state.response_times:
                    avg_time = np.mean(st.session_state.response_times)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                else:
                    st.metric("Avg Response Time", "N/A")
            
            with col3:
                if st.session_state.confidence_scores:
                    avg_conf = np.mean(st.session_state.confidence_scores)
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                else:
                    st.metric("Avg Confidence", "N/A")
            
            with col4:
                system_status = "🟢 Online" if not st.session_state.use_fallback else "🔄 Fallback"
                st.metric("System Status", system_status)
            
            # Display charts if plotly is available
            if PLOTLY_AVAILABLE and len(st.session_state.response_times) > 1:
                st.subheader("📈 Performance Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Response time chart
                    fig_time = px.line(
                        x=list(range(1, len(st.session_state.response_times) + 1)),
                        y=st.session_state.response_times,
                        title="Response Time Trend",
                        labels={'x': 'Query Number', 'y': 'Response Time (seconds)'}
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    # Confidence score chart
                    fig_conf = px.line(
                        x=list(range(1, len(st.session_state.confidence_scores) + 1)),
                        y=st.session_state.confidence_scores,
                        title="Confidence Score Trend",
                        labels={'x': 'Query Number', 'y': 'Confidence Score'}
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

    def show_query_history(self):
        """Display query history."""
        if st.session_state.query_history:
            st.subheader("📝 Recent Queries")
            
            # Show last 5 queries
            recent_queries = st.session_state.query_history[-5:]
            
            for i, query_data in enumerate(reversed(recent_queries), 1):
                with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {query_data['query'][:60]}..."):
                    st.write(f"**Question:** {query_data['query']}")
                    st.write(f"**Answer:** {query_data['answer'][:200]}...")
                    st.write(f"**Source:** {query_data.get('source_type', 'Unknown')}")
                    st.write(f"**Confidence:** {query_data.get('confidence', 0.0):.1%}")
                    st.write(f"**Time:** {query_data.get('timestamp', 'Unknown')}")

    def sidebar_configuration(self):
        """Display sidebar configuration options."""
        # st.sidebar.header("🔧 Configuration")
        
        # # System status
        # st.sidebar.subheader("System Status")
        
        # status_items = [
        #     ("KCC CSV Data", st.session_state.kcc_csv_loaded),
        #     ("Vector Store", st.session_state.vector_store_loaded),
        #     ("AI Model", st.session_state.llm_loaded),
        # ]
        
        # for item, status in status_items:
        #     icon = "✅" if status else "❌"
        #     st.sidebar.write(f"{icon} {item}")
        
        # st.sidebar.divider()
        
        # Query parameters
        st.sidebar.subheader("Query Parameters")
        
        similarity_threshold = st.sidebar.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum similarity score for vector search results"
        )
        
        top_k = st.sidebar.slider(
            "Max Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of similar documents to retrieve"
        )
        
        st.sidebar.divider()
        
        # Model selection (if available)
        if OFFLINE_LLM_AVAILABLE:
            st.sidebar.subheader("Model Selection")
            
            model_options = [
                "google/flan-t5-small",
                "google/flan-t5-base",
                
            ]
            
            selected_model = st.sidebar.selectbox(
                "AI Model",
                model_options,
                help="Select the AI model to use"
            )
            
            if st.sidebar.button("Reload Model"):
                st.session_state.llm_loaded = False
                self.load_llm_handler(selected_model)
                st.experimental_rerun()
        
        st.sidebar.divider()
        
        # Clear history button
        if st.sidebar.button("Clear History"):
            st.session_state.query_history = []
            st.session_state.total_queries = 0
            st.session_state.response_times = []
            st.session_state.confidence_scores = []
            st.success("History cleared!")
        
        return similarity_threshold, top_k

    def main_interface(self):
        """Main application interface."""
        # Header
        st.title("🌾 KCC Agricultural Q&A Assistant")
        st.markdown("*Offline-capable agricultural advisory system powered by KCC database*")
        
        # Show offline notice
        self.show_offline_notice()
        
        # Sidebar configuration
        similarity_threshold, top_k = self.sidebar_configuration()
        
        # Initialize systems
        if not st.session_state.system_ready:
            st.info("Initializing system components...")
            
            # Load KCC CSV data
            self.load_kcc_csv_data()
            
            # Load vector store
            self.load_vector_store()
            
            # Load LLM handler
            self.load_llm_handler()
            
            st.session_state.system_ready = True
            st.success("🎉 System ready! You can now ask agricultural questions.")
            st.experimental_rerun()
        
        # Main query interface
        st.subheader("💬 Ask Your Agricultural Question")
        
        # Sample questions
        sample_questions = [
            "How to control aphids in cotton crop?",
            "What is the best fertilizer for wheat?",
            "How to manage fungal diseases in tomato?",
            "When should I apply irrigation to rice?",
            "What are the symptoms of nitrogen deficiency?"
        ]
        
        with st.expander("💡 Sample Questions"):
            for i, question in enumerate(sample_questions, 1):
                if st.button(f"{i}. {question}", key=f"sample_{i}"):
                    st.session_state.current_query = question
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.get('current_query', ''),
            height=100,
            placeholder="e.g., How to control pests in cotton crop organically?"
        )
        
        # Process query
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Ask Question", type="primary"):
                if query.strip():
                    with st.spinner("Processing your question..."):
                        response = self.process_query(query, similarity_threshold, top_k)
                        
                        # Store in history
                        st.session_state.query_history.append(response)
                        st.session_state.total_queries += 1
                        
                        # Display response
                        self.display_response(response)
                        
                        # Clear current query
                        if 'current_query' in st.session_state:
                            del st.session_state.current_query
                else:
                    st.warning("Please enter a question.")
        
        with col2:
            if st.button("🗑️ Clear Query"):
                st.session_state.current_query = ""
                st.experimental_rerun()
        
        # Display statistics and history
        if st.session_state.total_queries > 0:
            tab1, tab2 = st.tabs(["📊 Statistics", "📝 History"])
            
            with tab1:
                self.show_statistics()
            
            with tab2:
                self.show_query_history()

    def run(self):
        """Run the Streamlit application."""
        try:
            self.main_interface()
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.info("Please refresh the page or contact support.")

# Main execution
if __name__ == "__main__":
    # Initialize and run the app
    app = OfflineKCCStreamlitApp()
    app.run()