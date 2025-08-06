import asyncio
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

async def _process_urls_in_background(urls: List[str]):
    """
    Helper function to run URL fetching, content extraction, and index updating
    in the background.
    """
    global all_chunks # Access the global list of chunks

    logging.info(f"Background task started for {len(urls)} URL(s).")
    new_extracted_chunks = await extract_content_from_urls(urls)
    
    if new_extracted_chunks:
        # Reload all chunks to ensure we have the latest state before appending
        # This prevents race conditions if multiple background tasks are running
        current_all_chunks = await load_chunks() 
        current_all_chunks.extend(new_extracted_chunks)
        await save_chunks(current_all_chunks)
        
        # Update the in-memory all_chunks for immediate use in the main loop
        # This is important for subsequent Q&A sessions in the same run
        all_chunks = current_all_chunks 

        logging.info("Processing all content chunks and updating the search index in background...")
        data_processor.create_and_save_index(all_chunks) # This will re-index all data
        logging.info("Background index update completed.")
    else:
        logging.warning("No new content was successfully extracted in background task.")


async def handle_question_answering(question: str):
    """
    Handles the Q&A process by searching the FAISS index and then using the LLM.
    """
    global llm_client # Declare global to modify the llm_client variable

    if not data_processor.index:
        print("The search index is not loaded. Please add content first to build the index.")
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
    
    # Load any existing data chunks at the start
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
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            urls_input = input("Enter URL(s) to process (comma-separated): ")
            urls = [url.strip() for url in urls_input.split(',')]
            if not urls:
                print("No URLs provided. Returning to main menu.")
                continue

            print(f"Starting background processing for {len(urls)} URL(s). You can continue using the system.")
            # Create a task to run the processing in the background
            asyncio.create_task(_process_urls_in_background(urls))

        elif choice == '2':
            if not all_chunks or not data_processor.index:
                print("The knowledge base is empty or the index is not loaded. Please add content by choosing option 1 first.")
                continue

            question = input("Enter your question: ")
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
    asyncio.run(main())