from typing import Dict
from typing_extensions import List, TypedDict, Tuple

from together import Together
import os

from food_asst_prompt import food_assistant_prompt
from retrieve_chunk_chroma import RetrieveChunkChroma
import re
from get_location_queries import GetLocationSubzone


class FoodRecommendationBot:
    def __init__(self, embded_model_name="BAAI/bge-large-en-v1.5",
                 llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
                 tool_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
                 client="together",
                 temperature=None,
                 print_source=False,
                 max_tokens=1024,
                 save_output=False,
                 n_first_lines=3,
                 vector_store=None,
                 max_num_full_history=5):
        self.embded_model_name = embded_model_name
        self.TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        self.temperature = temperature
        self.llm_model = llm_model
        self.tool_model = tool_model
        self.client = client
        self.print_source = print_source
        self.max_tokens = max_tokens
        self.save_output = save_output

        # Extract model name for file naming
        if self.llm_model == 'llama3.2':
            self.model_name = 'llama3.2'
        else:
            pattern = r'(gemma-\d+|Qwen\d+\.\d+-\d+B|Meta-Llama)'
            match = re.search(pattern, llm_model)
            self.model_name = match.group()

        # Initialize LLM and client
        self.llm = None
        self.client_endpoint = None
        if client == "ollama":
            self.llm = None
        elif client == "together":
            self.client_endpoint = Together()

        # Initialize embeddings and vector store
        self.vector_store = vector_store
        self.retrieve_class = RetrieveChunkChroma(self.vector_store, self.client_endpoint, self.embded_model_name,
                                                  n_first_lines=n_first_lines)

        self.query_history = []
        self.full_history = []
        self.max_num_full_history = max_num_full_history

        # TODO parameterise match_cutoff
        self.subzone_finder = GetLocationSubzone(area_file="area_to_subzone.json", subzone_file="sub_zone_nearby.json",
                                                 match_cutoff=0.6)

    def _rewrite_query(self, query: str, query_history: str) -> str:
        """Rewrite the query to better utilize the retrieval tool."""
        # Define the retrieval tool's capabilities
        tool_description = {
            "name": "retrieve",
            "description": "Retrieve information about restaurants from a vector database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant restaurant information [cuisine][type of place][location][any other relevant context]"
                    }
                },
                "required": ["query"]
            }
        }

        # Create a system message that explains the tool and its capabilities
        system_message = {
            "role": "system",
            "content": f"""You are a query rewriting assistant that helps optimize queries for semantic search in a vector database.

            The vector database contains restaurant summaries generated from Google Maps reviews. Each summary includes:
            - Restaurant name and location
            - Cuisine type and specialties
            - Price range and ambiance
            - Key highlights from reviews
            - Overall ratings and popularity

            Output format: [cuisine][type of place][location][any other relevant context]

            Given query history and current query, your task is to rewrite the user's query to be more effective for semantic search in this vector database following the output format. Focus on:
            1. For cuisine: If cuisine not mentioned, do not mention.
            2. For type of place: If type of place not mentioned, assume restaurants.
            3. For location: Get location from query or from chat context. DO NOT use prepositions
            4. Include any other relevant context mentioned by user

            Tool Description:
            {tool_description}

            IMPORTANT: Output ONLY the rewritten query. Do not include any explanations, additional text, or formatting. The output should be a single line containing just the rewritten query."""
        }

        # Create the user message with the query and context
        user_message = {
            "role": "user",
            "content": f"""Current query: {query}
            Query History: {query_history}

            Output only the rewritten query, nothing else."""
        }

        if self.client == "together":
            response = self.client_endpoint.chat.completions.create(
                model=self.tool_model,
                messages=[system_message, user_message],
                stream=False,
                max_tokens=200,
                temperature=0.3  # Lower temperature for more focused rewrites
            )
            rewritten_query = response.choices[0].message.content.strip()
        else:
            response = self.llm.invoke([system_message, user_message])
            rewritten_query = response.content.strip()

        return rewritten_query

    def _reformat_query(self, query):
        # Define the retrieval tool's capabilities
        system_message = {
            "role": "system",
            "content": f"""You are a query reformatting assistant that helps optimize queries for semantic search in a vector database.

        Given an input query, re-format it in the format shown below:
        search: [cuisine][type of place][any other relevant context], location: [location], search_more: [True / False]

        Given current query, your task is to rewrite the user's query to be more effective for semantic search in this vector database following the output format. Focus on:
        1. For cuisine: If cuisine not mentioned, do not mention.
        2. For type of place: If type of place not mentioned, assume restaurants or cafes.
        3. Include any other relevant context mentioned by user
        4. For location: Get location from query.
        5. For search_more, determine if user would like to find food places in surrounding areas. 
        For example, if user mentions "in", return False, if user mentions "near" or "around, return True

        IMPORTANT: Output ONLY the rewritten query. Do not include any explanations, additional text, or formatting. The output should be a single line containing just the rewritten query."""
        }
        # Create the user message with the query and context
        user_message = {
            "role": "user",
            "content": f"""Current query: {query}
                    Output only the rewritten query, nothing else."""
        }

        if self.client == "together":
            response = self.client_endpoint.chat.completions.create(
                model=self.tool_model,
                messages=[system_message, user_message],
                stream=False,
                max_tokens=200,
                temperature=0.3  # Lower temperature for more focused rewrites
            )
            reformat_query = response.choices[0].message.content.strip()
        else:
            response = self.llm.invoke([system_message, user_message])
            reformat_query = response.content.strip()

        return reformat_query

    def _generate(self, question: str, docs: List[Dict], chat_history: str):
        """Generate answer"""
        # docs_content = "\n\n".join(doc['text'] for doc in docs)
        docs_content = ""
        for doc in docs:
            distance = doc.get('distance', None)
            if distance is not None:
                docs_content += f"Estimated distance:{distance} km\n" + doc['text'] + "\n\n"
            else:
                docs_content += doc['text'] + "\n\n"
        source_list = "\n\nRetrieved the following places:\n"
        for doc in docs:
            source_list += (f"Name: {doc['place_name']} @ Zone: {doc['place_zone']}"
                            f"(Place ID: {doc['place_id']})\n")
            # print(f"Name: {doc['place_name']} @ Zone: {doc['place_zone']}"
            #               f"(Place ID: {doc['place_id']})\n")
        if self.save_output:
            with open('query.txt', "w") as f:
                f.write(question)

        system_prompt = (f"{food_assistant_prompt}\n"
                         f"### Context:\n"
                         f"{docs_content}\n"
                         f"Previous Messages:"
                         f"{chat_history}")
        num_words = len(system_prompt.split())
        # print(f"Num words system prompt: {num_words}")
        if self.save_output:
            with open('system_prompt.txt', "w") as f:
                f.write(system_prompt)

        if (self.llm_model == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K" or
                self.llm_model == "Qwen/Qwen2.5-7B-Instruct-Turbo" or self.llm_model == "llama3.2"):
            format_message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        else:
            full_prompt = (f"{system_prompt}\n"
                           f"### Question:\n"
                           f"{question}\n"
                           f"### Answer:  ")
            format_message = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        if self.client == "together":
            stream = self.client_endpoint.chat.completions.create(
                model=self.llm_model,
                messages=format_message,
                stream=True,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            for token in stream:
                if hasattr(token, 'choices') and token.choices[0].delta.content:
                    partial_answer = token.choices[0].delta.content
                    yield partial_answer
        else:
            response = self.llm.invoke(format_message)
            answer = response.content
            yield answer

        # Add source list at the end
        source_dict = {
            "sources": source_list
        }
        if self.print_source:
            yield source_dict

    def get_response(self, question: str, chat_history: List[dict]):
        """Get response for a given question"""
        # Update internal state history
        self.query_history.append(chat_history[-1])  # Get latest query only
        if len(chat_history) >= 2:
            for msg in chat_history[-2:]:
                self.full_history.append(msg)  # Get latest query and previous assistant reply
        # Remove first entry in history to maintain length
        if len(self.full_history) == self.max_num_full_history:
            self.full_history.pop(0)

        # Form strings
        query_history_str = "".join(f"{msg['role']}: {msg['content']}\n" for msg in self.query_history)
        full_history_str = "".join(f"{msg['role']}: {msg['content']}\n" for msg in self.full_history)
        # Rewrite the query
        rewritten_query = self._rewrite_query(question, query_history_str)
        # print(f"\nRewritten query: {rewritten_query}")
        reformat_query = self._reformat_query(rewritten_query)
        # print(f"\nReformat query: {reformat_query}")
        text_dict = {}
        full_text = ""
        for part in reformat_query.split(","):
            if part.strip():
                split_colon = part.strip().split(":")
                if len(split_colon) == 2:
                    category = split_colon[0]
                    text = split_colon[1].strip()
                else:
                    category = "search"
                    text = split_colon[0]
                text_dict[category] = text
                if category != "search_more":
                    full_text += text + " "

        get_nearby = text_dict.get('search_more') in ['True']
        location = text_dict.get('location', "")
        search = text_dict.get('search', "")

        place_id_set = set()
        all_docs = []
        if location:
            if get_nearby:
                nearby_subzones = self.subzone_finder.find_subzones(location, n_nearby=5)
            else:
                nearby_subzones = self.subzone_finder.find_subzones(location, n_nearby=2)
            # print(f"Found nearby zones: {nearby_subzones}")
            # if first entry has planning area, that means first entry is the subzone itself. Otherwise, there is no subzone to base distance measurement
            if nearby_subzones[0][2]:
                base_zone = nearby_subzones[0][0]
            else:
                base_zone = None
            for location in nearby_subzones:
                location_name = location[0]
                subzone_distance = location[1]
                planning_area = location[2]
                retrieve_query = search + " " + location_name
                if planning_area:
                    # If planning_area not None, means found location_name is a subzone so can filter
                    docs = self.retrieve_class.retrieve_and_join_chunks(retrieve_query, subzone=location_name,
                                                                        planning_area=planning_area, n_results=10)
                    # print(f"Found {len(docs)} docs with query: {retrieve_query}")
                else:
                    # If planning_area is None, means found location_name is direct from original query so no subzone
                    docs = self.retrieve_class.retrieve_and_join_chunks(retrieve_query, subzone=None,
                                                                        planning_area=None, n_results=20)
                    # print(f"Found {len(docs)} docs with query: {retrieve_query}")
                # Only add docs not in set
                for doc in docs:
                    place_id = doc["place_id"]
                    if place_id not in place_id_set:
                        place_id_set.add(place_id)
                        if subzone_distance is not None:
                            doc['distance'] = subzone_distance
                        else:
                            # For doc results queried without subzone info, estimate distance from doc subzone
                            if base_zone is not None:
                                doc_subzone = doc['place_zone']
                                distance = self.subzone_finder.subzone_distance(base_zone, doc_subzone)
                                if distance is not None:
                                    doc['distance'] = distance
                        # If base zone is known, only add docs that have distance meas
                        if base_zone is not None:
                            if "distance" in doc.keys():
                                all_docs.append(doc)
                        else:
                            all_docs.append(doc)
            if get_nearby:
                all_docs = sorted(all_docs, key=lambda doc: doc.get("distance", float("inf")))[:20]
            else:
                all_docs = sorted(all_docs, key=lambda doc: doc.get("distance", float("inf")))[:10]
        else:
            # If re-written query not formatted to output location, just use search to retrieve
            retrieve_query = full_text
            all_docs = self.retrieve_class.retrieve_and_join_chunks(retrieve_query, subzone=None,
                                                                    planning_area=None, n_results=10)

        # Stream the generation
        num_docs = len(all_docs)
        # print(f"Total number of docs: {num_docs}")
        full_answer = ""
        for response in self._generate(question, all_docs, full_history_str):
            if isinstance(response, str):
                full_answer += response
                if not self.save_output:
                    yield response
            else:
                full_answer += response.get("sources", "")
                if not self.save_output:
                    yield response

        if self.save_output:
            if not self.temperature:
                temperature = "None"
            else:
                temperature = str(self.temperature)
            with open(f'{self.model_name}_{temperature}_response.txt', "w") as f:
                f.write(full_answer)
            yield full_answer  # Yield the full answer when saving output

        return full_answer


# Example usage
if __name__ == "__main__":
    bot = FoodRecommendationBot()
    for response in bot.get_response("Suggest the best steak restaurants in Singapore", []):
        print(response, end="", flush=True) 