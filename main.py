import asyncio
from src.extractor import extract_content_from_urls
from src.storage import load_chunks, save_chunks
from src.processor import DataProcessor
from typing import List, Dict, Any

# Initialize the DataProcessor class
data_processor = DataProcessor()

async def handle_question_answering(question: str):
    """
    Handles the Q&A process by searching the FAISS index.
    """
    relevant_chunks = data_processor.search(question, k=3)
    if relevant_chunks:
        print("\n--- Relevant Information Found ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"Result {i+1} from URL: {chunk.get('url', 'N/A')}")
            # print(chunk['content'][:500] + "...") # Display first 500 characters
            print(f"Content: {chunk['content']}")
            print("-" * 20)
        print("----------------------------------\n")
    else:
        print("No relevant information found in the knowledge base.")

async def main():
    """
    The main asynchronous function for the interactive CLI.
    """
    # Load any existing data chunks at the start
    all_chunks = await load_chunks()
    
    # Try to load an existing FAISS index
    index_loaded = data_processor.load_index()
    if index_loaded:
        print("Existing FAISS index loaded successfully.")
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

            # Fetch and extract content from the new URLs
            print(f"Fetching and extracting content from {len(urls)} URL(s)...")
            new_chunks = await extract_content_from_urls(urls)
            
            if new_chunks:
                # Add the new chunks to our main list
                all_chunks.extend(new_chunks)
                # Save all chunks to the storage file
                await save_chunks(all_chunks)
                
                # Now, process all chunks and create/update the index
                print("Processing all content chunks and updating the search index...")
                data_processor.create_and_save_index(all_chunks)
                print("Index updated successfully.")
            else:
                print("No new content was successfully extracted.")

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