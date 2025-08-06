import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for file paths
FAISS_INDEX_FILE = "data/index.faiss"
CHUNK_METADATA_FILE = "data/chunks_with_metadata.json"
logging.getLogger("transformers").setLevel(logging.ERROR) 
logging.getLogger("sentence_transformers").setLevel(logging.ERROR) 
class DataProcessor:
    """
    Handles the embedding and indexing of content chunks.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name: The name of the sentence-transformer model to use.
        """
        logging.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Dict[str, Any]] = []

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates sentence embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A numpy array of embeddings.
        """
        #logging.info(f"Generating embeddings for {len(texts)} text chunks.")
        return self.model.encode(texts, convert_to_tensor=False)

    def _build_faiss_index(self, embeddings: np.ndarray):
        """
        Builds a FAISS index from a numpy array of embeddings.
        
        Args:
            embeddings: A numpy array of text embeddings.
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        #logging.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def _save_index_and_metadata(self, chunks: List[Dict[str, Any]]):
        """
        Saves the FAISS index and the corresponding chunks metadata to disk.
        
        Args:
            chunks: The list of chunks used to build the index.
        """
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        #logging.info(f"FAISS index saved to {FAISS_INDEX_FILE}")

        # Save the chunks with their metadata
        try:
            with open(CHUNK_METADATA_FILE, 'w') as f:
                json.dump(chunks, f, indent=4)
            #logging.info(f"Chunks metadata saved to {CHUNK_METADATA_FILE}")
        except Exception as e:
            logging.error(f"Error saving chunks metadata: {e}")

    def create_and_save_index(self, chunks: List[Dict[str, Any]]):
        """
        Orchestrates the creation and saving of the FAISS index.

        Args:
            chunks: A list of content chunks (dictionaries with 'content' key).
        """
        if not chunks:
            logging.warning("No chunks provided to create the index.")
            return

        texts_to_embed = [chunk['content'] for chunk in chunks]
        embeddings = self._generate_embeddings(texts_to_embed)
        self._build_faiss_index(embeddings)
        self._save_index_and_metadata(chunks)
        self.chunks = chunks
    
    def load_index(self):
        """
        Loads the FAISS index and the chunks metadata from disk if they exist.
        """
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_METADATA_FILE):
            try:
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                #logging.info(f"FAISS index loaded from {FAISS_INDEX_FILE}")
                
                with open(CHUNK_METADATA_FILE, 'r') as f:
                    self.chunks = json.load(f)
                #logging.info(f"Loaded {len(self.chunks)} chunks from {CHUNK_METADATA_FILE}")
                return True
            except Exception as e:
                #logging.error(f"Error loading FAISS index or metadata: {e}")
                return False
        return False
    
    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for the query text.
        
        Args:
            query_text: The text query to search for.
            k: The number of top results to return.

        Returns:
            A list of the top k most relevant chunks.
        """
        if not self.index:
            logging.warning("FAISS index is not loaded. Cannot perform search.")
            return []

        logging.info(f"Searching for '{query_text}'...")
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        
        return results
