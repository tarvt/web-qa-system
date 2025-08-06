import asyncio
import concurrent.futures # <--- NEW IMPORT
from src.extractor import extract_content_from_urls
from src.storage import load_chunks, save_chunks
from src.processor import DataProcessor
from src.llm_api import GeminiLLM
from typing import List, Dict, Any
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the DataProcessor and GeminiLLM classes
data_processor = DataProcessor()
llm_client = None # Initialize as None, will be set up when needed

# Declare all_chunks as a global variable to be accessible and modifiable
# by both main and background processing functions.
all_chunks: List[Dict[str, Any]] = []

# Create a ThreadPoolExecutor for running blocking I/O operations (like input())
# This allows the asyncio event loop to remain responsive.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # Use 1 worker for sequential input

async def async_input(prompt: str = "") -> str:
    """
    Asynchronously gets user input by running the blocking input() in a separate thread.
    """
    return await asyncio.get_running_loop().run_in_executor(executor, input, prompt)


async def _process_urls_in_background(urls: List[str]):
    """
    Helper function to run URL fetching, content extraction, and index updating
    in the background.
    """
    global all_chunks # Explicitly declare intent to modify the global all_chunks

    #logging.info(f"Background task started for {len(urls)} URL(s).")
    new_extracted_chunks = await extract_content_from_urls(urls)
    
    if new_extracted_chunks:
        # Load the absolute latest state of chunks from disk
        # This is crucial to avoid overwriting changes from other potential background tasks
        # or missed updates if the main loop's `all_chunks` isn't fully synchronized yet.
        current_disk_chunks = await load_chunks() 
        #logging.info(f"Background: Loaded {len(current_disk_chunks)} chunks from disk before adding new.")
        current_disk_chunks.extend(new_extracted_chunks)
        await save_chunks(current_disk_chunks)
        
        # Update the global all_chunks variable
        # This ensures the main loop's in-memory data is consistent with disk
        all_chunks = current_disk_chunks 
        #logging.info(f"Background: Global all_chunks updated to {len(all_chunks)} chunks.")

        #logging.info("Processing all content chunks and updating the search index in background...")
        # Pass the globally updated 'all_chunks' to the data processor
        data_processor.create_and_save_index(all_chunks) 
        #logging.info("Background index update completed.")
    else:
        logging.warning("No new content was successfully extracted in background task.")


async def handle_question_answering(question: str):
    """
    Handles the Q&A process by searching the FAISS index and then using the LLM.
    """
    global llm_client # Declare global to modify the llm_client variable

    # Ensure the latest chunks are loaded before attempting Q&A
    # This is important if a background task just finished and updated the disk.
    global all_chunks
    #logging.info("Q&A: Reloading all_chunks from disk to ensure latest data.")
    all_chunks = await load_chunks() 
    #logging.info(f"Q&A: Loaded {len(all_chunks)} chunks from disk for current query.")
    
    # Also ensure the data_processor's internal chunks are up-to-date for searching
    # This is a critical step to ensure the search is performed on the latest index
    if not data_processor.index or data_processor.index.ntotal != len(all_chunks):
        #logging.info("Q&A: Index not loaded or out of sync. Attempting to rebuild/reload index.")
        data_processor.create_and_save_index(all_chunks) # Rebuilds if out of sync or not loaded
        data_processor.load_index() # Ensures the index is loaded into memory after potential rebuild


    # Use the global all_chunks to check if there's data
    if not all_chunks or not data_processor.index:
        print("The knowledge base is empty or the index is not loaded. Please add content by choosing option 1 first.")
        return

    if llm_client is None:
        try:
            llm_client = GeminiLLM()
        except ValueError as e:
            print(f"Error initializing LLM: {e}")
            print("Please ensure your GEMINI_API_KEY environment variable is set correctly.")
            return

    print(f"Searching for relevant information for: '{question}'...")
    relevant_chunks = data_processor.search(question, k=3) # Retrieve top 3 relevant chunks

    if relevant_chunks:
        print("Found relevant information. Generating answer with AI...")
        # Pass the question and relevant chunks to the LLM
        answer = await llm_client.generate_answer(question, relevant_chunks)
        
        print("\n--- AI Generated Answer ---")
        print(answer)
        print("---------------------------\n")

        # Optionally, show the sources
        print("\n--- Sources Used ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"{i+1}. {chunk.get('url', 'N/A')}")
        print("--------------------\n")

    else:
        print("No relevant information found in the knowledge base to answer your question.")
        print("Consider adding more diverse content related to your question.")


async def main():
    """
    The main asynchronous function for the interactive CLI.
    """
    global all_chunks # Declare global to modify the all_chunks variable
    
    # Load any existing data chunks at the start into the global variable
    all_chunks = await load_chunks()
    
    # Try to load an existing FAISS index
    index_loaded = data_processor.load_index()
    if index_loaded:
        print("Existing FAISS index and metadata loaded successfully.")
    else:
        print("No existing FAISS index found. An index will be created when you add content.")

    print(f"Welcome to the Web Content QA System!")
    print(f"Loaded {len(all_chunks)} existing content chunks from the data/ directory.")

    while True:
        print("\nWhat would you like to do?")
        print("1. Add new URL(s) to the knowledge base.")
        print("2. Ask a question.")
        print("3. Exit.")
        
        # Use the asynchronous input function and await its result before stripping
        choice = (await async_input("Enter your choice (1, 2, or 3): ")).strip() # <--- MODIFIED LINE

        if choice == '1':
            # Use the asynchronous input function and await its result before stripping
            urls_input = (await async_input("Enter URL(s) to process (comma-separated): ")).strip() # <--- MODIFIED LINE
            urls = [url.strip() for url in urls_input.split(',')]
            if not urls:
                print("No URLs provided. Returning to main menu.")
                continue

            print(f"Scheduling background processing for {len(urls)} URL(s). You can continue using the system.")
            # This is the critical line: create the task and let it run.
            asyncio.create_task(_process_urls_in_background(urls))
            print("Processing task scheduled. Check logs for progress. Returning to menu.") # Immediate feedback

        elif choice == '2':
            # Check against the global all_chunks
            if not all_chunks or not data_processor.index:
                print("The knowledge base is empty or the index is not loaded. Please add content by choosing option 1 first.")
                continue

            # Use the asynchronous input function and await its result before stripping
            question = (await async_input("Enter your question: ")).strip() # <--- MODIFIED LINE
            if not question:
                print("No question provided. Returning to main menu.")
                continue

            await handle_question_answering(question)

        elif choice == '3':
            print("Thank you for using the system. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Ensure the executor is shut down cleanly when the program exits
        executor.shutdown(wait=True)