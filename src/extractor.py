import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Any, Optional


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """
    Fetches the content of a single URL asynchronously.
    Args:
        session: The aiohttp client session.
        url: The URL to fetch.

    Returns:
        The HTML content as a string, or None if there was an error.
    """
    try:
        # Use a User-Agent header to mimic a web browser and avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        async with session.get(url, headers=headers, timeout=10) as response:
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            #logging.info(f"Successfully fetched URL: {url}")
            return await response.text()
    except aiohttp.ClientError as e:
        logging.error(f"aiohttp error fetching {url}: {e}")
    except asyncio.TimeoutError:
        logging.error(f"Timeout error fetching {url}")
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {e}")
    return None

def extract_text_from_html(html_content: str) -> str:
    """
    Extracts and cleans meaningful text from HTML content.
    
    Args:
        html_content: The raw HTML content as a string.

    Returns:
        A cleaned string of meaningful text.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted tags that don't contain meaningful text
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()

        # Find main content sections. This is a heuristic and may need adjustment.
        # Prioritize articles, main content, or specific semantic tags.
        main_content = soup.find('main') or soup.find('article')
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to getting text from the body if main/article tags are not found
            text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''

        # Use a regex to clean up extra whitespace and newlines
        clean_text = re.sub(r'\s+', ' ', text).strip()

        return clean_text
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {e}")
        return ""

async def extract_content_from_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Main function to orchestrate the fetching and extraction process for multiple URLs.
    Args:
        urls: A list of URLs to process.
    Returns:
        A list of dictionaries, where each dictionary contains the original URL and the extracted text.
    """
    extracted_data = []
    
    # Create a single aiohttp session to manage connections efficiently
    async with aiohttp.ClientSession() as session:
        # Create a list of tasks for fetching each URL
        tasks = [fetch_url(session, url) for url in urls]
        
        # Run all fetch tasks concurrently
        html_contents = await asyncio.gather(*tasks)

        # Process the results
        for url, html_content in zip(urls, html_contents):
            if html_content:
                text = extract_text_from_html(html_content)
                if text:
                    # Create a structured dictionary for the extracted data
                    extracted_data.append({
                        "url": url,
                        "content": text
                    })
                    #logging.info(f"Extracted content from {url}, length: {len(text)} characters.")
                else:
                    logging.warning(f"Could not extract meaningful text from {url}")
            else:
                logging.warning(f"Skipping {url} due to a fetch error.")

    return extracted_data
