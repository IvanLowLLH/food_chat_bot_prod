from typing import Dict
from typing_extensions import List, TypedDict

from together import Together
import os

from food_asst_prompt import food_assistant_prompt
from retrieve_chunk_chroma import RetrieveChunkChroma
import re

class FoodRecommendationBot:
    def __init__(self, embded_model_name="BAAI/bge-large-en-v1.5",
                 llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
                 client="together",
                 temperature=None,
                 print_source=False,
                 max_tokens=1024,
                 save_output=False,
                 n_first_lines=3,
                 vector_store=None):
        self.embded_model_name = embded_model_name
        self.TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        self.temperature = temperature
        self.llm_model = llm_model
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
        self.retrieve_class = RetrieveChunkChroma(self.vector_store, self.client_endpoint, self.embded_model_name, n_first_lines=n_first_lines)

    def _rewrite_query(self, query: str, chat_history: List[dict]) -> str:
        """Rewrite the query to better utilize the retrieval tool."""
        chat_context = ""
        if chat_history:
            chat_context = "\nPrevious conversation:\n"
            for msg in chat_history[-3:]:  # Only include last 3 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_context += f"{role}: {msg['content']}\n"

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

Your task is to rewrite the user's query to be more effective for semantic search in this vector database following the output format. Focus on:
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
            "content": f"""Original query: {query}
            Context: {chat_context}
            
            Output only the rewritten query, nothing else."""
        }

        if self.client == "together":
            response = self.client_endpoint.chat.completions.create(
                model=self.llm_model,
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
    
    def _generate(self, question: str, docs: List[Dict], chat_history: List[dict]):
        """Generate answer"""
        docs_content = "\n\n".join(doc['text'] for doc in docs)
        source_list = "\n\nRetrieved the following places:\n"
        for doc in docs:
            source_list += (f"Name: {doc['place_name']} @ Zone: {doc['place_zone']}"
                          f"(Place ID: {doc['place_id']})\n")
        
        # Format chat history
        chat_context = ""
        if chat_history:
            chat_context = "\nPrevious conversation:\n"
            for msg in chat_history[-3:]:  # Only include last 3 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_context += f"{role}: {msg['content']}\n"

        if self.save_output:
            with open('query.txt', "w") as f:
                f.write(question)
        
        system_prompt = (f"{food_assistant_prompt}\n"
                      f"### Context:\n"
                      f"{docs_content}\n"
                      f"Previous Messages:"
                      f"{chat_context}")
        num_words = len(system_prompt.split())
        print(f"Num words system prompt: {num_words}")
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
        # Rewrite the query
        rewritten_query = self._rewrite_query(question, chat_history)
        print(f"\nRewritten query: {rewritten_query}")
        
        # Get relevant documents using rewritten query
        docs = self.retrieve_class.retrieve_and_join_chunks(rewritten_query, n_results=10)
        
        # Stream the generation
        full_answer = ""
        for response in self._generate(question, docs, chat_history):
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