# Web Content Smart Extraction and Question Answering System

## Project Description
The goal of this project is to design and implement a system that can take a website URL (e.g., a documentation page or a blog post), extract and structure its content, and store it in a way that can be used to answer textual questions in later stages. This system will feature an interactive command-line interface (CLI) for users to add content and ask questions.

---

## Project Stages:
1.  **Content Extraction**
    - The system receives multiple URLs.
    - It extracts the textual content, focusing on sections containing useful information, using **aiohttp** and **BeautifulSoup**.
    - The extracted content is saved in a structured format (`data/chunks.jsonl`).

2.  **Data Preparation for Q&A**
    - Data is processed to be searchable and retrievable based on semantic similarity.
    - Text content is converted into numerical vectors (embeddings) using **sentence-transformers**.
    - An efficient vector index is built using **FAISS** for fast similarity search.
    - The process is asynchronous, allowing the user to continue interacting while content is being processed.

3.  **Question Answering**
    - The system receives a textual question from the user.
    - It finds the most relevant section(s) from the stored and indexed data using **FAISS**.
    - The retrieved relevant text is then used as context for a Large Language Model (LLM) API (e.g., Google's Gemini API) to generate a concise and intelligent answer, which is displayed to the user.

---

### Important Notes:
- The choice of tools, methods, and database design is up to the developer.
- The focus is on problem analysis, proper system design, clean coding, and good documentation.
- A full user interface is not required; a simple command-line interface (CLI) script is sufficient.

---

## Folder Structure
````
web-content-qa-system/
├── src/
│   ├── init.py         # Initializes the Python package
│   ├── extractor.py        # Handles web scraping logic (fetching & parsing)
│   ├── processor.py        # Handles data preparation (embeddings, FAISS indexing)
│   ├── storage.py          # Handles data persistence (saving/loading chunks)
│   ├── llm_api.py          # (Future) Handles interaction with the LLM API
│   └── cli.py              # (Future) Main interactive CLI logic (currently in main.py)
├── data/                   # Directory to store persistent data
│   ├── chunks.jsonl        # Stores raw extracted text chunks
│   ├── index.faiss         # The FAISS vector index file
│   └── chunks_with_metadata.json # Metadata for chunks linked to FAISS index
├── requirements.txt        # Lists all project dependencies
├── .gitignore              # Specifies files/folders to be ignored by Git
├── README.md               # This project description and instructions
└── main.py                 # The main entry point for the interactive CLI
````

---

## Technologies and Tools
**Programming Language:** Python

### Core Libraries:
- **aiohttp**: Asynchronous HTTP client for efficient web fetching.
- **BeautifulSoup4**: HTML parsing for content extraction.
- **sentence-transformers**: For generating semantically rich text embeddings.
- **faiss-cpu**: For high-performance similarity search on text embeddings.
- **google-generativeai** (Future): Python client library for interacting with Google's Gemini API.

**User Interface:** Interactive Command-Line Interface (CLI).

---

## How to Run
Follow these steps to set up and run the system:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd web-content-qa-system
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    Make sure your `requirements.txt` file contains:
    ```
    aiohttp==3.9.5
    beautifulsoup4==4.12.3
    faiss-cpu==1.8.0
    sentence-transformers==2.7.0
    # google-generativeai (will be added in the next stage)
    ```

4.  **Create Data Directory:**
    ```bash
    mkdir data
    ```

5.  **Run the CLI Application:**
    ```bash
    python main.py
    ```

    The CLI will then prompt you to choose between adding URLs or asking questions.