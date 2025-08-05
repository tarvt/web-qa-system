import json
import os
import asyncio
from typing import List, Dict, Any

CHUNK_FILE_PATH = "data/chunks.jsonl"

async def save_chunks(chunks: List[Dict[str, Any]]):
    """
    Asynchronously saves a list of content chunks to a JSON Lines file.
    Each chunk is a dictionary with at least 'url' and 'content' keys.

    Args:
        chunks: The list of dictionaries to save.
    """
    os.makedirs(os.path.dirname(CHUNK_FILE_PATH), exist_ok=True)
    try:
        with open(CHUNK_FILE_PATH, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        print(f"Successfully saved {len(chunks)} chunks to {CHUNK_FILE_PATH}")
    except Exception as e:
        print(f"Error saving chunks: {e}")

async def load_chunks() -> List[Dict[str, Any]]:
    """
    Asynchronously loads content chunks from a JSON Lines file.

    Returns:
        A list of content chunks, or an empty list if the file doesn't exist.
    """
    chunks = []
    if not os.path.exists(CHUNK_FILE_PATH):
        print("Chunk file not found. Starting with an empty knowledge base.")
        return chunks
    
    try:
        with open(CHUNK_FILE_PATH, "r") as f:
            for line in f:
                chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} chunks from {CHUNK_FILE_PATH}")
    except Exception as e:
        print(f"Error loading chunks: {e}")
        chunks = []
        
    return chunks