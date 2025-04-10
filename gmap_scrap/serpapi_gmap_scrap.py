import json
from serpapi import GoogleSearch
import sqlite3

import os
from dotenv import load_dotenv
import geopandas as gpd
from functions import get_subzone

load_dotenv()
serpApi_key = os.getenv('SERPAPI_KEY')

# Replace with your JSON file path
area_json_filename = "sg_area_centroids.json"
# Open the JSON file and load its content
with open(area_json_filename, "r", encoding="utf-8") as file:
    area_data = json.load(file)

# Load the GeoJSON file
geojson_path = "sg_subzones.geojson"  # Replace with your file path
gdf = gpd.read_file(geojson_path)

cuisines_in_singapore = [
    # Southeast Asian Cuisines
    "South East Asian / Peranakan",
    # East Asian Cuisines
    "Japanese / Korean",
    # South Asian Cuisines
    "Indian",
    # Middle Eastern & Mediterranean Cuisines
    "South Asian / Middle Eastern",
    # European Cuisines
    "European", "French", "Italian", "Swiss",
    # American & Latin American Cuisines
    "Latin American",
    # African Cuisines
    "African",
    # Other Cuisines
    "Vegetarian & Vegan Cuisine"
]
place_type_list = [cuisine +" restaurants" for cuisine in cuisines_in_singapore]
place_type_list.extend(["bar", "cafe", "dessert"])
detail_dir = "details"
log_file = "log.txt"

os.makedirs(detail_dir, exist_ok=True)

# Create/connect to an SQLite database
conn = sqlite3.connect("food_places.db")
cursor = conn.cursor()

# Initialize SQLite database
cursor.execute("""
    CREATE TABLE IF NOT EXISTS places (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        place_id TEXT,
        name TEXT,
        address TEXT,
        area TEXT,
        sub_zone TEXT,
        type TEXT,
        rating REAL,
        num_reviews REAL,
        detail_path TEXT,
        review_path TEXT,
        summary_path TEXT
        summary_long_path TEXT
    )
""")
conn.commit()

# Function to fetch restaurant data
def fetch_food_places(query, lat, lon, start=0):
    params = {
        "engine": "google_maps",
        "q": query,
        "type": "search",
        "hl": "en",
        "gl": "sg",
        "api_key": serpApi_key,
        "ll": f"@{lat},{lon},15z",  # Singapore's latitude & longitude
        "start": start
    }

    search = GoogleSearch(params)
    return search.get_dict()

def place_exists(place_id):
    """Check if a place already exists in the database."""
    cursor.execute("SELECT 1 FROM places WHERE place_id = ?", (place_id,))
    return cursor.fetchone() is not None

def save_to_db(place_id, name, address, area, sub_zone, type, rating, num_reviews,
               detail_path, review_path, summary_path, summary_long_path):
    """Save a new place to the database if not a duplicate."""
    cursor.execute("INSERT INTO places (place_id, name, address, area, sub_zone, type, rating, num_reviews, detail_path, review_path, summary_path, summary_long_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (place_id, name, address, area, sub_zone, type, rating, num_reviews, detail_path, review_path, summary_path, summary_long_path))
    conn.commit()

for area in area_data:
    for place_type in place_type_list:
        places_count = 0
        area_name = area["name"]
        area_lat = area["latitude"]
        area_lon = area["longitude"]

        gmap_query = f"{place_type} in {area_name} Singapore"
        for start in range(0, 60, 20):
            print("------------------------------------------------")
            print(f"Searching for {gmap_query} with start: {start} ")
            print("------------------------------------------------")
            try:
                gmap_results = fetch_food_places(gmap_query, area_lat, area_lon, start)
            except Exception as e:
                print(f"Error fetching results for {gmap_query} at start {start}: {e}")
                continue
            if "local_results" in gmap_results:
                for place in gmap_results["local_results"]:

                    place_id = place.get("place_id", "Null")
                    name = place.get("title", "Null")
                    address = place.get("address", "Null")
                    rating = place.get("rating", 0)
                    num_reviews = place.get("reviews", 0)
                    type = place.get("type", "Null")
                    place_lat = place.get("gps_coordinates", {}).get("latitude", None)
                    place_lon = place.get("gps_coordinates", {}).get("longitude", None)
                    sub_zone = get_subzone(gdf, place_lat, place_lon)

                    print(f"Processing {name} in {address}")
                    # If place_id not in database
                    if not place_exists(place_id) and sub_zone:
                        no_pass_bool = False
                        print(f"{name} in {address} not in database. Adding to database.")
                        places_count += 1
                        # Save place details to json file
                        detail_json_filename = f"{place_id}.json"
                        detail_path = f"{detail_dir}/{detail_json_filename}"
                        with open(detail_path, "w", encoding="utf-8") as file:
                            json.dump(place, file, indent=4, ensure_ascii=False)

                        # Save information and paths to database
                        # placeholder values
                        review_path = "None"
                        summary_path = "None"
                        summary_long_path = None
                        save_to_db(place_id, name, address, area_name, sub_zone, type, rating, num_reviews,
                                   detail_path, review_path, summary_path, summary_long_path)
                        print(f"Done processing for {name}.")
                    else:
                        print(f"{name} in {address} in database. Skipping.")
            else:
                print(f"No results found for {gmap_query}. Moving to next search.")
                break

            print(f"Done searching for {gmap_query} with start: {start}. Sleep for 3s")
            # time.sleep(3)

        log_txt = f"Found a total of {places_count} places from {gmap_query}"
        print(f"Log: {log_txt}")
        with open(log_file, "a") as file:
            file.write((log_txt + "\n"))






