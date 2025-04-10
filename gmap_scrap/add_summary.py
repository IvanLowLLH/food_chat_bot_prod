"""
Find and save summaries of places in database
"""
import sqlite3
from dotenv import load_dotenv
from functions import open_json
from together import Together
from summary_prompts import summary_prompt
from multiprocessing import Pool, cpu_count

load_dotenv()

summary_tokens = 2048
summary_dir = "summaries_long"
# os.mkdir(summary_dir)
summary_llm_model = "google/gemma-2-9b-it"
client_endpoint = Together()

def process_place(row):
    """Process a single place and generate its summary."""
    # Extract place information
    _, place_id, place_name, address, place_area, place_zone, _, rating, _, detail_path, review_path, _, _ = row
    print(f"Processing {place_id}")
    
    detail_data = open_json(detail_path)
    place_types = "".join(place_type + "," for place_type in detail_data["types"])
    gmap_mrt_station = detail_data.get('gmap_results', {}).get('MRT/Subway Station', "")
    gmap_shopping_mall = detail_data.get('gmap_results', {}).get('Shopping Mall', "")
    onemap_buildings = detail_data.get('onemap_building', "")
    building_names = ''.join(building + ", " for building in onemap_buildings)
    reviews = open_json(review_path)

    full_review_text = (
        f"Name of place: {place_name}. Located at {place_area} area and {place_zone} zone in Singapore.\n"
        f"Building Names: {building_names}\n"
        f"Nearest MRT/subway: {gmap_mrt_station}\n"
        f"Nearest Shopping Mall: {gmap_shopping_mall}\n"
        f"Full address: {address}.\n"
        f"Overall rating of {rating}.\n"
        f"It is classified as {place_types}. \n")

    # Extract information from reviews
    print(f'Extracting reviews for {place_id}')
    review_count = 0
    for review in reviews:
        review_count += 1
        review_text = review.get("snippet", "None")
        review_rating = review.get("rating", 0)
        review_rating_details = review.get("details", {})

        combined_text = (f"Google Map review {review_count}\n"
                         f"Reviewer gave {review_rating} rating\n"
                         f"Reviewer gave following details for rating\n")
        for key, value in review_rating_details.items():
            combined_text += f"{key}: {value}\n"
        combined_text += f"Review text: {review_text}\n\n"
        full_review_text += combined_text

    summary_prompt_template = f"""
    {summary_prompt}

    Context containing information and reviews:
    {full_review_text}

    Answer:"""
    
    print(f'Sending summary prompt into LLM for {place_id}')
    # Sending full prompt into model
    format_message = [
        {
            "role": "user",
            "content": summary_prompt_template
        }
    ]
    
    try:
        stream = client_endpoint.chat.completions.create(
            model=summary_llm_model,
            messages=format_message,
            stream=False,
            max_tokens=summary_tokens,
        )
    except:
        # In case max_tokens too large
        stream = client_endpoint.chat.completions.create(
            model=summary_llm_model,
            messages=format_message,
            stream=False,
            max_tokens=1024,
        )
    
    summary_response = stream.choices[0].message.content
    print(f'Saving summary response into file for {place_id}')
    
    # Create summary file
    summary_json_filename = f"{place_id}_summary.txt"
    summary_path = f"{summary_dir}/{summary_json_filename}"
    with open(summary_path, "w") as file:
        file.write(summary_response)

    # Update database
    conn = sqlite3.connect("food_places.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE places SET summary_long_path = ? WHERE place_id = ?", (summary_path, place_id))
    conn.commit()
    conn.close()
    
    print(f'Saved file and added to database for {place_id}!')
    return place_id

def main():
    """Main function to process places in parallel."""
    # Create/connect to an SQLite database
    conn = sqlite3.connect("food_places.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM places
        WHERE review_path != 'None'
        AND summary_long_path IS NULL
    """)
    results = cursor.fetchall()
    conn.close()
    
    # Determine number of processes (use 75% of available CPU cores)
    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_processes} processes")
    
    # Process places in parallel
    with Pool(processes=num_processes) as pool:
        processed_places = pool.map(process_place, results)
    
    print(f"Processed {len(processed_places)} places")

if __name__ == "__main__":
    main()
