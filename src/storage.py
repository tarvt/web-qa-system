import json
import os
import asyncio
from typing import List, Dict, Any
import logging # <--- Ensure logging is imported here

# Set up basic logging (if not already done, though main.py sets it up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_FILE_PATH = "data/chunks.jsonl"
FAISS_INDEX_FILE = "data/index.faiss" # Added for consistency in logging
CHUNK_METADATA_FILE = "data/chunks_with_metadata.json" # Added for consistency in logging

async def save_chunks(chunks: List[Dict[str, Any]]):
    """
    Asynchronously saves a list of content chunks to a JSON Lines file.
    Each chunk is a dictionary with at least 'url' and 'content' keys.

    Args:
        chunks: The list of dictionaries to save.
    """
    absolute_path = os.path.abspath(CHUNK_FILE_PATH)
    #logging.info(f"Attempting to save {len(chunks)} chunks to: {absolute_path}")
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True) # Use absolute path for mkdir
    try:
        with open(absolute_path, "w") as f: # Use absolute path for open
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        #logging.info(f"Successfully saved {len(chunks)} chunks to {absolute_path}")
    except Exception as e:
        logging.error(f"Error saving chunks to {absolute_path}: {e}")

async def load_chunks() -> List[Dict[str, Any]]:
    """
    Asynchronously loads content chunks from a JSON Lines file.

    Returns:
        A list of content chunks, or an empty list if the file doesn't exist.
    """
    chunks = []
    absolute_path = os.path.abspath(CHUNK_FILE_PATH)
    #logging.info(f"Attempting to load chunks from: {absolute_path}")

    if not os.path.exists(absolute_path): # Use absolute path for exists check
        logging.info(f"Chunk file not found at {absolute_path}. Starting with an empty knowledge base.")
        return chunks
    
    try:
        with open(absolute_path, "r") as f: # Use absolute path for open
            for line in f:
                chunks.append(json.loads(line))
        #logging.info(f"Loaded {len(chunks)} chunks from {absolute_path}")
    except Exception as e:
        logging.error(f"Error loading chunks from {absolute_path}: {e}")
        chunks = [] # Ensure chunks is empty on error
        
    return chunks
