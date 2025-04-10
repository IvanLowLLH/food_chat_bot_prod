import sqlite3
import pickle
import json
from serpapi import GoogleSearch

import os
from dotenv import load_dotenv

load_dotenv()
serpApi_key = os.getenv('SERPAPI_KEY')

num_pages = 2 # Number of pages of review to extract
review_dir = "reviews"

# Add reviews for selected place_ids
place_ids_file = "filered_place_ids.pkl"
with open(place_ids_file, "rb") as file:
    filter_place_ids = pickle.load(file)

# Create/connect to an SQLite database
conn = sqlite3.connect("food_places.db")
cursor = conn.cursor()

def fetch_reviews(place_id, num_pages=2):
    reviews_list = []
    seen_reviews = set()  # Track unique review snippets
    next_page_token = None  # Token for pagination
    for page in range(0, num_pages):
        params = {
            "engine": "google_maps_reviews",
            "hl": "en",
            "gl": "sg",
            "place_id": place_id,
            "api_key": serpApi_key,
            "limit": 10  # Fetch 10 reviews per request
        }
        if next_page_token:
            params["next_page_token"] = next_page_token  # Add token for next page

        search = GoogleSearch(params)
        results = search.get_dict()

        if "reviews" not in results or not results["reviews"]:
            print("No more reviews found. Stopping pagination.")
            break  # Stop if no new reviews are found

        new_reviews = 0  # Track how many new reviews are added

        for review in results["reviews"]:
            review_id = review.get("snippet")  # Use review snippet as unique ID
            if review_id and review_id not in seen_reviews:
                seen_reviews.add(review_id)
                reviews_list.append(review)
                new_reviews += 1

        if new_reviews == 0:
            print("No new unique reviews found. Stopping pagination.")
            break  # Stop if only duplicates are found

        try:
            next_page_token = results["serpapi_pagination"]["next_page_token"]
        except:
            next_page_token = None
        if not next_page_token:
            print("No next_page_token available. Stopping pagination.")
            break  # Stop if there are no more pages

    return reviews_list

total_place_ids = len(filter_place_ids)
count = 0
ignore_review_path = False
for place_id in filter_place_ids:
    count += 1
    print("--------------------------")
    print(f"({count}/{total_place_ids}) Processing {place_id}. Checking if review_path exist.")
    print("--------------------------")
    cursor.execute("SELECT review_path, rating, num_reviews FROM places WHERE place_id = ?", (place_id,))
    result = cursor.fetchone()

    review_path, rating, num_reviews = result
    if num_reviews <= 20:
        print("Too little reviews.")
        continue
    if rating < 3:
        print("Rating too low.")
        continue
    if review_path == "None" or ignore_review_path:
        print(f"{place_id} does not have review_path. Finding reviews now.")
        # Find reviews and add to json file
        place_reviews = fetch_reviews(place_id, num_pages=num_pages)
        num_reviews = len(place_reviews)
        if num_reviews > 0:
            print(f"Found {num_reviews} reviews for {place_id}.")
            reviews_json_filename = f"{place_id}_reviews.json"
            review_path = f"{review_dir}/{reviews_json_filename}"
            with open(review_path, "w", encoding="utf-8") as file:
                json.dump(place_reviews, file, indent=4, ensure_ascii=False)
            print(f"Wrote reviews into {review_path}")
            # Add review_path to database
            cursor.execute("UPDATE places SET review_path = ? WHERE place_id = ?", (review_path, place_id))
            conn.commit()
            print(f"Added {place_id} review_path to database.")
        else:
            print("No reviews found. Not adding to database")
    else:
        print(f"{place_id} already has review_path: {review_path}")

conn.close()