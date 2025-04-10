from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import sqlite3
import pickle
from tqdm import tqdm
import nltk

nltk.download('punkt_tab')
DB_PATH = "food_places.db"  # Path to your SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
    SELECT *
    FROM places
    WHERE summary_long_path != 'None'
""")
rows = cursor.fetchall()
conn.close()
doc_list = []
doc_info_list = []
for row in tqdm(rows, desc="Processing places"):
    _, place_id, place_name, address, place_area, place_zone, place_type, rating, _, detail_path, review_path, _, summary_path = row
    with open(summary_path, "r", encoding="utf-8") as file:
        text = file.read()
    doc_text = word_tokenize(text.lower())
    doc_info = {
        'place_id': place_id,
        'place_name': place_name
    }
    doc_list.append(doc_text)
    doc_info_list.append(doc_info)

# Initialize BM25
bm25 = BM25Okapi(doc_list)

#To save bm25 object
with open('rank_bm25result_k50', 'wb') as bm25result_file:
    # pickle.dump(bm25, bm25result_file)
    pickle.dump({"bm25": bm25, "doc_infos": doc_info_list}, bm25result_file)