"""
Code to filter places to add reviews for
"""
from collections import defaultdict
import sqlite3
import csv
import pickle

num_place_per_zone = 20

conn = sqlite3.connect('food_places.db')
cursor = conn.cursor()
# Find restaurants with rating greater or equal 4 stars
cursor.execute("""
    SELECT *
    FROM places
    WHERE type LIKE '%restaurant%'
    AND rating >= 4
    AND review_path == 'None'
""")

results = cursor.fetchall()
print(f"Total number of places: {len(results)}")
places_by_area = defaultdict(list)
for row in results:
    places_by_area[row[4]].append(row)  # Assuming 'area' is at index 4

# Sort places within each area by rating * num_reviews
for area, places in places_by_area.items():
    places.sort(key=lambda x: x[7] * x[8], reverse=True)  # Index 6: rating, Index 7: num_reviews
    places_by_area[area] = places[:num_place_per_zone]  # Keep top N places per area

# Write to CSV
filter_place_id = []
with open("filter_places.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "place_id", "name", "address", "area", "sub_zone", "type", "rating",
                     "num_reviews", "detail_path", "review_path", "summary_path"])

    for area, places in places_by_area.items():
        writer.writerows(places)
        for place in places:
            filter_place_id.append(place[1])

with open("filered_place_ids.pkl", "wb") as file:
    pickle.dump(filter_place_id, file)