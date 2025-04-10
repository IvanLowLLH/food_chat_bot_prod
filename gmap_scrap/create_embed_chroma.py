"""
Create or update Chroma database from embedded summaries
"""
import sqlite3
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from together import Together
import os
import re
from tqdm import tqdm
import uuid

load_dotenv()
# Configuration
DB_PATH = "food_places.db"  # Path to your SQLite database
CHROMA_PATH = "chroma_gmapfood"  # Path to Chroma database
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")  # Set your Together API key
client = Together()

model_name = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 440 # 512 - 72 tokens (for 1st 3 lines)
TOKEN_RATIO = 1.5  # 1 word ≈ 1.5 tokens
MAX_WORDS = int(MAX_TOKENS / TOKEN_RATIO)  # Maximum words per chunk
n_first_lines = 3 # Lines containing Name, Location and Type information

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
try:
    collection = chroma_client.get_collection("gmap_food")
except:
    collection = chroma_client.create_collection("gmap_food")

def get_embeddings(text: str) -> list[float]:
    """Get embeddings for a text using Together API."""
    response = client.embeddings.create(
        input=text,
        model=model_name
    )
    return response.data[0].embedding

def split_into_words(text: str) -> list[str]:
    """Split text into words while preserving punctuation and spacing."""
    # Use regex to split on whitespace while preserving punctuation
    return re.findall(r'\S+|\s+', text)

def join_words(words: list[str]) -> str:
    """Join words back into text while preserving spacing."""
    return ''.join(words)

def extract_first_lines(text: str, num_lines: int = 3) -> tuple[str, str]:
    """Extract first n lines from text and return them along with the remaining text."""
    lines = text.split('\n')
    first_lines = '\n'.join(lines[:num_lines])
    remaining_text = '\n'.join(lines[num_lines:])
    return first_lines, remaining_text

def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """
    Split text into chunks of approximately max_words each.
    Preserves original formatting and tries to break at natural boundaries.
    First 3 lines are preserved in each chunk.
    """
    # Extract first 3 lines
    first_lines, remaining_text = extract_first_lines(text, num_lines=n_first_lines)

    # Split remaining text into words while preserving spacing
    words = split_into_words(remaining_text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for word in words:
        # If adding this word would exceed max_words, start a new chunk
        if current_word_count >= max_words and current_chunk:
            # Add first lines to the chunk
            chunk_text = first_lines + '\n' + join_words(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_word_count = 0

        current_chunk.append(word)
        current_word_count += 1

    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = first_lines + '\n' + join_words(current_chunk)
        chunks.append(chunk_text)

    return chunks

def get_existing_place_ids():
    """Fetch existing place_ids from Chroma to avoid duplicates."""
    try:
        results = collection.get(include=['metadatas'])
        return set([metadata["place_id"] for metadata in results['metadatas']])
    except:
        return set()

def process_place(place_id: str, place_name: str, address: str, place_area: str, 
                 place_zone: str, place_type: str, rating: float, summary_path: str) -> tuple[bool, str]:
    """Process a single place and add its chunks to Chroma."""
    try:
        with open(summary_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Split text into chunks
        chunks = chunk_text(text)
        print(f"Split text into {len(chunks)} chunks")
        
        # Store each chunk in Chroma
        text_chunks = []
        embedding_chunks = []
        meta_data_chunks = []
        # id_chunks = []

        for chunk_idx, chunk in enumerate(chunks):
            # Get embeddings for the chunk
            embedding = get_embeddings(chunk)
            
            # Store in Chroma with chunk information
            meta_data = {
                "place_id": place_id,
                "place_name": place_name,
                "address": address,
                "place_area": place_area,
                "place_zone": place_zone,
                "place_type": place_type,
                "rating": rating,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks)
            }
            text_chunks.append(chunk)
            embedding_chunks.append(embedding)
            meta_data_chunks.append(meta_data)
            # id_chunks.append(f"{place_id}_chunk_{chunk_idx}")

        id_chunks = [str(uuid.uuid4()) for _ in text_chunks]
        collection.add(
            documents=text_chunks,
            embeddings=embedding_chunks,
            metadatas=meta_data_chunks,
            ids=id_chunks
        )
        print(f"Added {len(chunks)} chunks  for {place_id} to Chroma.")
            
        return True, f"Successfully processed {place_id} with {len(chunks)} chunks"
    except Exception as e:
        return False, f"Error processing {place_id}: {str(e)}"

def main():
    # Get existing place IDs
    existing_place_ids = get_existing_place_ids()
    print(f"Found {len(existing_place_ids)} existing places in Chroma")
    
    # Get all places from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM places
        WHERE summary_long_path != 'None'
    """)
    rows = cursor.fetchall()
    conn.close()

    total_places = len(rows)
    print(f"Processing {total_places} places...")

    successes = []
    failures = []

    # Process each place sequentially with a progress bar
    for row in tqdm(rows, desc="Processing places"):
        _, place_id, place_name, address, place_area, place_zone, place_type, rating, _, detail_path, review_path, _, summary_path = row
        
        print(f"\nProcessing {place_id} ({place_name})")
        
        if place_id in existing_place_ids:
            print(f"Skipping {place_id}, already in Chroma.")
            successes.append(f"Skipped {place_id}, already in Chroma.")
            continue

        success, message = process_place(
            place_id, place_name, address, place_area, 
            place_zone, place_type, rating, summary_path
        )
        
        if success:
            successes.append(message)
        else:
            failures.append(message)
            print(f"Error: {message}")

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successes)} places")
    if failures:
        print(f"Failed to process {len(failures)} places:")
        for failure in failures:
            print(failure)

if __name__ == "__main__":
    main()
