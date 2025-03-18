import streamlit as st
from llm_gmap import FoodRecommendationBot
import time
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, UTC
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import pytz

import chromadb
from chromadb.config import Settings
from together import Together

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREBASE_CREDS = "firebase_key.json"

# Timezone settings
SGT = pytz.timezone('Asia/Singapore')

# Firestore database settings
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDS)
    firebase_admin.initialize_app(cred)

# Connect to Firestore
db = firestore.client()

# LLM & embedding settings
llm_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"
tool_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"
embed_model_name = "BAAI/bge-large-en-v1.5"
bm25_file = "rank_bm25result_k50"
n_first_lines = 3

# Rate limiting settings
COOLDOWN_SECONDS = 2  # Time between queries
MAX_QUERIES_PER_HOUR = 30  # Maximum queries per hour
QUERY_WINDOW_HOURS = 1  # Time window for query counting
TRACK_IP = True

client = Together()
vector_store = chromadb.PersistentClient(
    path="chroma_bge_large_gmapfood_long_14Mar",
    settings=Settings(anonymized_telemetry=False)
).get_collection("gmap_food")


def get_client_ip():
    """Get client IP address using an external API."""
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        return response.json().get("ip", "unknown")
    except:
        return "unknown"


def load_ip_tracking(client_ip):
    """Load IP tracking data from Firestore."""
    doc_ref = db.collection("ip_tracking").document(client_ip)
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {"last_query_time": None, "query_history": []}


def save_ip_tracking(client_ip, tracking_data):
    """Save IP tracking data to Firestore."""
    db.collection("ip_tracking").document(client_ip).set(tracking_data)


def save_query_to_firebase(session_id, query, full_response, timestamp):
    """Save query details to Firestore."""
    # db.collection("queries").add({
    #     "ip": ip,
    #     "query": query,
    #     "timestamp": timestamp
    # })
    doc_ref = db.collection("queries_new").document(session_id)
    # Create a dictionary for the new query
    new_query = {
        "query": query,
        "response": full_response,
        "timestamp": timestamp
    }
    # Use arrayUnion to add the new query to the 'queries' array
    doc_ref.set({
        "queries": firestore.ArrayUnion([new_query])
    }, merge=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bot" not in st.session_state:
        st.session_state.bot = FoodRecommendationBot(
            embded_model_name=embed_model_name,
            llm_model=llm_model,
            bm25_file=bm25_file,
            vector_store=vector_store,
            n_first_lines=n_first_lines,
            save_output=False
        )
    if "ip_tracking" not in st.session_state:
        client_ip = get_client_ip()
        st.session_state.ip_tracking = load_ip_tracking(client_ip)


def get_current_time():
    """Get current time in Singapore timezone."""
    return datetime.now(SGT)


def parse_timestamp(timestamp_str):
    """Parse ISO timestamp string to datetime object in Singapore timezone."""
    if not timestamp_str:
        return datetime.min.replace(tzinfo=SGT)
    try:
        dt = datetime.fromisoformat(timestamp_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(SGT)
    except ValueError:
        return datetime.min.replace(tzinfo=SGT)


def can_make_query():
    """Check if a new query can be made based on rate limits."""
    current_time = get_current_time()

    ip_data = st.session_state.ip_tracking
    last_query_time = parse_timestamp(ip_data["last_query_time"])

    # Check cooldown period
    time_since_last_query = (current_time - last_query_time).total_seconds()
    if time_since_last_query < COOLDOWN_SECONDS:
        return False, f"Please wait {COOLDOWN_SECONDS - int(time_since_last_query)} seconds before making another query."

    # Clean up old queries from history
    cutoff_time = current_time - timedelta(hours=QUERY_WINDOW_HOURS)
    ip_data["query_history"] = [
        time for time in ip_data["query_history"]
        if parse_timestamp(time) > cutoff_time
    ]

    # Check hourly limit
    if len(ip_data["query_history"]) >= MAX_QUERIES_PER_HOUR:
        next_query_time = parse_timestamp(ip_data["query_history"][0]) + timedelta(hours=QUERY_WINDOW_HOURS)
        wait_time = next_query_time - current_time
        minutes = int(wait_time.total_seconds() / 60)
        return False, f"Query limit reached for your IP. Please wait {minutes} minutes before making another query."

    return True, ""


def update_query_tracking():
    """Update query tracking after a successful query."""
    current_time = get_current_time()
    client_ip = get_client_ip()

    # Update IP tracking
    st.session_state.ip_tracking["last_query_time"] = current_time.isoformat()
    st.session_state.ip_tracking["query_history"].append(current_time.isoformat())

    # Save tracking data to Firebase
    save_ip_tracking(client_ip, st.session_state.ip_tracking)


# Initialize session state
initialize_session_state()
if "session_start_id" not in st.session_state:
    st.session_state.session_start_id = str(int(datetime.now(UTC).timestamp()))
session_start_id = st.session_state.session_start_id
# Set page config
st.set_page_config(
    page_title="Food Recommendation Chatbot",
    page_icon="🍽️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .stMarkdown {
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🍽️ Singapore Food Recommendation Chatbot")
st.markdown("Ask me about food places in Singapore!  \n"
            "For best results, try to specify a location like 'Cafes in Bugis' or 'Italian restaurants in city hall'  \n")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar Rate Limit Info
with st.sidebar:
    st.title("Chat Controls")
    if st.button("Clear Chat History & Restart"):
        st.session_state.messages = []
        st.session_state.bot.query_history = []
        st.session_state.bot.full_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Usage Limits")
    client_ip = get_client_ip()
    queries_remaining = MAX_QUERIES_PER_HOUR - len(st.session_state.ip_tracking["query_history"])
    st.markdown(f"Queries remaining for you: **{queries_remaining}**")

    if st.session_state.ip_tracking["query_history"]:
        reset_time = parse_timestamp(st.session_state.ip_tracking["query_history"][0]) + timedelta(hours=QUERY_WINDOW_HOURS)
        st.markdown(f"Next reset at: **{reset_time.strftime('%I:%M:%S %p')} SGT**")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This chatbot helps you find the food places in Singapore based on Google Maps reviews and ratings.

        Ask questions like:
        - "What are the best steak restaurants in Singapore?"
        - "Suggest Japanese restaurants in Bedok"
        - If you see a restaurant you are interested in: "Tell me more about [restaurant name]"

        If seem to be stuck in a loop of recommending same places, press Clear Chat History.

        Disclaimer: AIs are known to hallucinate so please check before going!
        """)

# Chat input
if prompt := st.chat_input("What kind of food are you looking for?"):
    # Check rate limits
    if TRACK_IP:
        can_query, message = can_make_query()
        if not can_query:
            st.error(message)
            st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            for response in st.session_state.bot.get_response(prompt, st.session_state.messages):
                if isinstance(response, str):
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)  # Small delay for smoother streaming
                else:
                    full_response += response.get("sources", "")
                    message_placeholder.markdown(full_response)

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Save query to Firebase
    current_time = get_current_time()
    save_query_to_firebase(session_start_id, prompt, full_response, current_time)

    # Update query tracking
    update_query_tracking()