import os
import google.generativeai as genai
import logging
from typing import List, Dict, Any, Optional

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiLLM:
    """
    Handles interactions with the Google Gemini LLM API.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the Gemini LLM client.
        Expects the GEMINI_API_KEY environment variable to be set.
        
        Args:
            model_name: The name of the Gemini model to use (e.g., "gemini-1.5-flash", "gemini-pro").
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please get your API key from Google AI Studio (https://aistudio.google.com/app/apikey) "
                "and set it as an environment variable."
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logging.info(f"Gemini LLM initialized with model: {model_name}")

    async def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer to a question using the LLM, given context chunks.

        Args:
            question: The user's question.
            context_chunks: A list of relevant content chunks (dictionaries with 'content' key).

        Returns:
            The generated answer from the LLM.
        """
        if not context_chunks:
            return "I couldn't find any relevant information in the knowledge base to answer your question."

        # Construct the prompt for the LLM
        context_text = "\n\n".join([chunk['content'] for chunk in context_chunks])
        
        prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the answer is not available in the context, state that you don't have enough information.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        logging.info(f"Sending prompt to Gemini LLM for question: '{question}'")
        try:
            # Use the non-streaming generateContent API
            # Implement exponential backoff for API calls
            retries = 0
            max_retries = 5
            base_delay = 1 # seconds

            while retries < max_retries:
                try:
                    response = await asyncio.to_thread(self.model.generate_content, prompt)
                    # Check if response has candidates and content
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        return response.candidates[0].content.parts[0].text
                    else:
                        logging.warning(f"Gemini API returned an empty or malformed response: {response}")
                        return "I received an empty response from the AI model."
                except genai.types.BlockedPromptException as e:
                    logging.error(f"Prompt was blocked by safety settings: {e.response.prompt_feedback}")
                    return "I'm sorry, your request was blocked due to safety concerns."
                except Exception as e:
                    logging.error(f"Error calling Gemini API: {e}")
                    retries += 1
                    delay = base_delay * (2 ** retries)
                    logging.info(f"Retrying in {delay} seconds (retry {retries}/{max_retries})...")
                    await asyncio.sleep(delay)
            
            logging.error(f"Failed to get a response from Gemini API after {max_retries} retries.")
            return "I'm sorry, I could not get a response from the AI model after multiple attempts."

        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM generation: {e}")
            return "An unexpected error occurred while generating the answer."

