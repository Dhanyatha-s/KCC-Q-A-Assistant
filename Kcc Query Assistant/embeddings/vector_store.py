import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, vector_store_dir: str = "vector_store"):
        """
        Initialize vector store with FAISS for efficient similarity search.
        """
        self.vector_store_dir = vector_store_dir
        self.index = None
        self.chunks = []
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
    def load_vector_store(self):
        """
        Load embeddings, chunks, and create FAISS index.
        """
        print("Loading vector store...")
        
        # Load model info
        model_info_path = os.path.join(self.vector_store_dir, "model_info.json")
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        self.embedding_dim = model_info['embedding_dim']
        model_name = model_info['model_name']
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load embeddings
        embeddings_path = os.path.join(self.vector_store_dir, "embeddings.npy")
        embeddings = np.load(embeddings_path)
        print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        
        # Load chunks
        chunks_path = os.path.join(self.vector_store_dir, "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Vector store loaded successfully!")
        print(f"Index size: {self.index.ntotal}")
        
    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: User query text
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.index is None or self.model is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= threshold:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarity)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        print(f"Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict[str, Any]:
        """Get a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk['id'] == chunk_id:
                return chunk
        return None

class ChromaDBStore:
    """
    Alternative implementation using ChromaDB (if preferred over FAISS)
    """
    def __init__(self, vector_store_dir: str = "vector_store"):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=os.path.join(vector_store_dir, "chromadb"))
            self.collection = None
            self.model = None
        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
    
    def load_vector_store(self):
        """Load or create ChromaDB collection."""
        print("Loading ChromaDB vector store...")
        
        # Load model info
        model_info_path = os.path.join("vector_store", "model_info.json")
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        model_name = model_info['model_name']
        self.model = SentenceTransformer(model_name)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="kcc_knowledge_base",
            metadata={"description": "KCC agricultural knowledge base"}
        )
        
        # Check if collection is empty and populate if needed
        if self.collection.count() == 0:
            self._populate_collection()
        
        print(f"ChromaDB loaded with {self.collection.count()} documents")
    
    def _populate_collection(self):
        """Populate ChromaDB with embeddings and chunks."""
        print("Populating ChromaDB collection...")
        
        # Load chunks
        chunks_path = os.path.join("vector_store", "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Prepare data for ChromaDB
        ids = [str(chunk['id']) for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add to collection (ChromaDB will generate embeddings)
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} documents to ChromaDB")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        formatted_results = []
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0], 
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            formatted_results.append({
                'id': int(doc_id),
                'text': document,
                'metadata': metadata,
                'similarity_score': 1 - distance,  # Convert distance to similarity
                'rank': i + 1
            })
        
        return formatted_results

def test_vector_store():
    """Test function to verify vector store functionality."""
    print("🧪 Testing Vector Store...")
    
    # Test FAISS implementation
    vs = VectorStore()
    vs.load_vector_store()
    
    test_queries = [
        "paddy pest control Tamil Nadu",
        "wheat fertilizer management",
        "tomato disease prevention",
        "drought management groundnut"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        results = vs.search_similar(query, top_k=3, threshold=0.3)
        
        if results:
            for result in results:
                print(f"  Rank {result['rank']}: {result['similarity_score']:.3f} - {result['query'][:60]}...")
        else:
            print(" No relevant results found")
    
    print("\n Vector store test completed!")

if __name__ == "__main__":
    test_vector_store()