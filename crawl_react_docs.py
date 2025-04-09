import os
import sys
import json
import asyncio
import requests
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from openai import AsyncOpenAI
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from supabase import create_client, Client
 
load_dotenv()
 
# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
 
url_filter = URLPatternFilter(patterns=["*learn*", "*reference*"])
 
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
 
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)
 
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
 
        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
 
        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
 
        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break
 
        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1
 
        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
 
        # Move start position for next chunk
        start = max(start + 1, end)
 
    return chunks
 
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
   
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}
 
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error
 
async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
   
    # Get embedding
    embedding = await get_embedding(chunk)
   
    # Create metadata
    metadata = {
        "source": "react_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
   
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )
 
async def insert_chunk(chunk: ProcessedChunk):
    """Insert or update a processed chunk in Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
       
        # Use upsert instead of insert to overwrite existing data
        result = supabase.table("react_pages").insert(data).execute()
        print(f"Upserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error upserting chunk: {e}")
        return None
 
async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Extra safety check - don't process empty content
    if not markdown or markdown.strip() == "":
        print(f"Skipping empty content for {url}")
        return
       
    # Split into chunks
    chunks = chunk_text(markdown)
   
    # Skip if no valid chunks
    if not chunks:
        print(f"No valid chunks found for {url}")
        return
   
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
   
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)
 
async def crawl_react_docs(url: str, max_depth: int = 3):
    """Crawl the React documentation starting from the given URL."""
   
    # Create a filter chain with URL patterns
    # filter_chain = FilterChain([
    #     URLPatternFilter(patterns=[
    #         "*blog*",
    #         "*core*",
    #         "*advanced*",
    #         "*api*",
    #         "*extraction*"
    #     ])
    # ])
   
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False  # Apply the filter chain here
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )
 
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url=url,
            config=config
        )
       
        # Rest of your function remains the same
        if not results:
            print(f"No results returned for: {url}")
            return
           
        print(f"Crawled {len(results)} pages starting from {url}")
       
        # Process each result in the list
        for result in results:
            # Check for 404 explicitly, along with other failures
            if hasattr(result, 'status_code') and result.status_code == 404:
                page_url = getattr(result, 'url', 'unknown URL')
                print(f"Skipping 404 Not Found: {page_url}")
                continue
               
            if hasattr(result, 'success') and result.success:
                page_url = getattr(result, 'url', url)
                print(f"Successfully crawled: {page_url}")
                await process_and_store_document(page_url, result.markdown)
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                status_code = getattr(result, 'status_code', 'unknown status')
                page_url = getattr(result, 'url', 'unknown URL')
                print(f"Failed to crawl: {page_url} - Status: {status_code} - Error: {error_msg}")
 
async def main():
    start_url = "https://react.dev/"  # Starting point
    await crawl_react_docs(start_url)
 
if __name__ == "__main__":
    asyncio.run(main())