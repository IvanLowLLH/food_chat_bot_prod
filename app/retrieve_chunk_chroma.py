"""
Retrieve and join chunks from Chroma database
"""
from typing import List, Dict
from collections import defaultdict


class RetrieveChunkChroma:
    def __init__(self, vector_store, client, model_name, n_first_lines: int = 3):
        self.n_first_lines = n_first_lines
        self.vector_store = vector_store
        self.client = client
        self.model_name = model_name

    def _extract_first_lines(self, text: str, num_lines: int = 3) -> tuple[str, str]:
        """Extract first n lines from text and return them along with the remaining text."""
        lines = text.split('\n')
        first_lines = '\n'.join(lines[:num_lines])
        remaining_text = '\n'.join(lines[num_lines:])
        return first_lines, remaining_text

    def _get_embeddings(self, text: str) -> list[float]:
        """Get embeddings for a text using Together API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

    def _get_all_chunks_for_place(self, place_id: str) -> List[Dict]:
        """Get all chunks for a specific place_id."""
        try:
            # Query Chroma for all chunks with the given place_id (returns documents, metadatas)
            results = self.vector_store.get(
                where={"place_id": place_id},
            )

            # Sort chunks by chunk_index
            chunks = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                chunks.append({
                    'text': doc,
                    'metadata': metadata
                })

            sorted_chunks = sorted(chunks, key=lambda x: x['metadata']['chunk_index'])

            return sorted_chunks
        except Exception as e:
            print(f"Error getting chunks for place {place_id}: {e}")
            return []

    def retrieve_and_join_chunks(self, query: str, subzone: str | list = None, planning_area: str = None, n_results: int = 5) -> List[Dict]:
        """
        Search for relevant chunks and join them by place_id.
        Returns a list of dictionaries containing joined text and metadata for each place.
        """
        try:
            # Get query embedding
            query_embedding = self._get_embeddings(query)

            # Search in Chroma. Returns documents, metadata, distances
            filter_dict = None
            if subzone:
                if isinstance(subzone, str):
                    filter_dict = {
                        'place_zone': subzone
                    }
                elif isinstance(subzone, list):
                    if len(subzone) == 1:
                        filter_dict = {
                            'place_zone': subzone[0]
                        }
                    else:
                        # Include multiple subzone in filter search
                        filter_list = []
                        for zone in subzone:
                            filter_list.append({'place_zone': zone})
                        filter_dict = {"$or": filter_list}
            if subzone and planning_area:
                filter_dict = {
                    "$or":[
                        {
                            'place_zone': subzone
                        },
                        {
                            'place_area': planning_area
                        }
                    ]
                }
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                where=filter_dict
            )

            # Group chunks by place_id
            place_chunks = defaultdict(list)
            for doc, metadata, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                place_id = metadata['place_id']
                place_chunks[place_id].append({
                    'text': doc,
                    'metadata': metadata,
                    'score': score
                })

            # For each place found, get all its chunks
            joined_results = []
            for place_id, initial_chunks in place_chunks.items():
                # Get all chunks for this place
                all_chunks = self._get_all_chunks_for_place(place_id)

                # Get the best score from initial search
                best_score = min(chunk['score'] for chunk in initial_chunks)

                # Join all chunks for this place
                joined_text = ""
                for chunk in all_chunks:
                    first_lines, remaining_text = self._extract_first_lines(chunk['text'], self.n_first_lines)
                    if chunk['metadata']['chunk_index'] == 0:
                        joined_text += first_lines + "\n"
                    joined_text += remaining_text

                # Get place info from first chunk
                place_info = all_chunks[0]['metadata']

                joined_results.append({
                    'place_id': place_id,
                    'place_name': place_info['place_name'],
                    'rating': place_info['rating'],
                    'place_zone': place_info['place_zone'],
                    'place_area': place_info['place_area'],
                    'text': joined_text,
                    'score': best_score,
                    'num_chunks': len(all_chunks),
                    'metadata': place_info
                })

            # Sort results by rating
            joined_results.sort(key=lambda x: x['score'])
            return joined_results[:n_results]

        except Exception as e:
            print(f"Error in retrieve_and_join_chunks: {e}")
            return []