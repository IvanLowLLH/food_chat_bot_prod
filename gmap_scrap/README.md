# Creating vector and lexical database of food places

## 0. API Tokens
This codes uses SerpAPI to extract Google Maps and TogetherAI to summarise. Get their API tokens
and save to a `.env` file.
SerpAPI API should be saved as `SERPAPI_KEY` and TogtherAI token saved as `TOGETHER_API_KEY`.

## 1. Create SQLite3 database of food places
Use `serapi_gmap_scarp.py` to create SQLite3 database of food places
by using SerpAPI to scrap Google Maps for restaurants/bars/cafes
in different areas of Singapore.

## 2. Filter places to add reviews for
Extracting reviews uses SerpAPI available searches very quickly so
you want to find reviews for only some places.

Use `filter_places.py`. Can edit the SQL query to filter for places.
`num_place_per_zone` limits the total number of places per Planning Area.

## 3. Extract reviews
`add_reviews.py` uses SerpAPI to extract Google Map reviews. For each place,
2 pages of reviews, which is about 18 reviews, will be extracted. Edit `place_ids_file`
for the pickle output of `filter_places.py`.

## 4. Generate summaries
`add_summary.py` uses TogetherAI and specifically Gemma2 9B to generate summaries of places that
have extracted reviews.

## 5. Create vector database
Use `create_embed_chroma.py` to generate embeddings using TogetherAI `bge-large-en-v1.5` model
and then add the embeddings to a Chroma database

## 6. Create BM25 data file
Use `rankBM25_generation.py` to generate BM25 data file