import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from tqdm import tqdm
import json
from typing import List, Dict, Any

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def load_processed_data(self, csv_path: str, max_rows: int = 2500) -> pd.DataFrame:
        """
        Load your preprocessed KCC CSV data with optional row limit.
        
        Args:
            csv_path: Path to the CSV file
            max_rows: Maximum number of rows to load (default: 2500)
        
        Expected columns: 'id', 'query', 'response', 'state', 'crop', 'category', etc.
        """
        print(f"Loading processed data from: {csv_path}")
        
        # Load the data with row limit
        df = pd.read_csv(csv_path, nrows=max_rows)
        
        # Ensure required columns exist
        required_cols = ['QueryText', 'KccAns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Loaded {len(df)} records (limited to {max_rows} rows)")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def sample_data_strategically(self, df: pd.DataFrame, n_samples: int = 2500) -> pd.DataFrame:
        """
        Sample data strategically to get diverse representation.
        
        This ensures we get good coverage across different categories, states, crops etc.
        """
        print(f"Sampling {n_samples} rows strategically...")
        
        # If we already have fewer rows than requested, return all
        if len(df) <= n_samples:
            print(f"Dataset has {len(df)} rows, which is <= {n_samples}. Using all data.")
            return df
        
        # Try to sample proportionally across categories if they exist
        sampled_dfs = []
        
        # Check if we have categorical columns to stratify by
        categorical_cols = []
        for col in ['category', 'state', 'crop']:
            if col in df.columns and df[col].notna().sum() > 0:
                categorical_cols.append(col)
        
        if categorical_cols:
            # Stratified sampling based on the first available categorical column
            primary_col = categorical_cols[0]
            print(f"Stratified sampling based on '{primary_col}' column")
            
            # Get value counts and calculate proportional samples
            value_counts = df[primary_col].value_counts()
            
            for category, count in value_counts.items():
                # Calculate how many samples from this category
                proportion = count / len(df)
                category_samples = max(1, int(proportion * n_samples))
                
                # Sample from this category
                category_df = df[df[primary_col] == category]
                if len(category_df) <= category_samples:
                    sampled_dfs.append(category_df)
                else:
                    sampled_dfs.append(category_df.sample(n=category_samples, random_state=42))
            
            # Combine all sampled data
            sampled_df = pd.concat(sampled_dfs, ignore_index=True)
            
            # If we have too many, randomly sample down to exact number
            if len(sampled_df) > n_samples:
                sampled_df = sampled_df.sample(n=n_samples, random_state=42)
                
        else:
            # Simple random sampling if no categorical columns
            print("No categorical columns found. Using random sampling.")
            sampled_df = df.sample(n=n_samples, random_state=42)
        
        print(f"Sampled {len(sampled_df)} rows")
        
        # Show distribution if categorical column exists
        if categorical_cols:
            print(f"\nSample distribution by {categorical_cols[0]}:")
            print(sampled_df[categorical_cols[0]].value_counts().head(10))
        
        return sampled_df.reset_index(drop=True)
    
    def create_document_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame rows into document chunks for embedding.
        
        Each chunk combines query and response for better semantic search.
        """
        chunks = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating chunks"):
            # Combine query and response for embedding
            combined_text = f"Query: {row['QueryText']} Answer: {row['KccAns']}"

            chunk = {
                'id': idx,
                'text': combined_text,
                'query': row['QueryText'],
                'response': row['KccAns'],
                'metadata': {
                    'state': row.get('StateName', 'Unknown'),
                    'district': row.get('DistrictName', 'Unknown'),
                    'block': row.get('BlockName', 'Unknown'),
                    'sector': row.get('Sector', 'Unknown'),
                    'crop': row.get('Crop', 'Unknown'),
                    'category': row.get('Category', 'General'),
                    'query_type': row.get('QueryType', 'General'),
                    'created_on': row.get('CreatedOn', 'Unknown'),
                    'original_id': row.get('document', idx)
                }
            }
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} document chunks")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for all document chunks.
        """
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} texts...")
        print(f"Using model: {self.model_name}")
        print(f"Estimated time: ~{len(texts) // batch_size * 2} seconds")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings_and_chunks(self, chunks: List[Dict[str, Any]], 
                                  embeddings: np.ndarray, 
                                  output_dir: str = "vector_store"):
        """
        Save embeddings and chunks to disk for later use.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to: {embeddings_path}")
        
        # Save chunks metadata
        chunks_path = os.path.join(output_dir, "chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks to: {chunks_path}")
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'num_chunks': len(chunks),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Saved model info to: {info_path}")
        
        return embeddings_path, chunks_path, info_path

def main():
    # Configuration
    CSV_PATH = "Kcc_processed_data.csv"  # Update with your CSV path
    OUTPUT_DIR = "vector_store"
    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and good quality
    MAX_ROWS = 2500  # Limit for faster processing
    
    # Alternative models:
    # "all-mpnet-base-v2"  # Higher quality, slower
    # "multi-qa-MiniLM-L6-cos-v1"  # Optimized for Q&A
    
    try:
        # Initialize embedding generator
        embed_gen = EmbeddingGenerator(model_name=MODEL_NAME)
        
        # Load your processed data (with row limit)
        df = embed_gen.load_processed_data(CSV_PATH, max_rows=MAX_ROWS)
        
        # Optionally do strategic sampling for even better diversity
        # df = embed_gen.sample_data_strategically(df, n_samples=MAX_ROWS)
        
        # Create document chunks
        chunks = embed_gen.create_document_chunks(df)
        
        # Generate embeddings
        embeddings = embed_gen.generate_embeddings(chunks)
        
        # Save everything
        embed_gen.save_embeddings_and_chunks(chunks, embeddings, OUTPUT_DIR)
        
        print("\n✅ Embedding generation completed successfully!")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"📊 Total embeddings: {len(embeddings)}")
        print(f"🔢 Embedding dimension: {embeddings.shape[1]}")
        print(f"⏱️  Processing time was much faster with {MAX_ROWS} rows!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the CSV file: {CSV_PATH}")
        print("Please update the CSV_PATH variable with the correct path to your processed data.")
    except Exception as e:
        print(f"❌ Error during embedding generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()