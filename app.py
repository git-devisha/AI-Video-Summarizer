import inspect
# Fix for deprecated getargspec in Python 3.11+
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import streamlit as st
import google.generativeai as genai
from google.generativeai import upload_file, get_file
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import tempfile
import time
import os
from pathlib import Path

# --- Load API Key from .env ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video AI Summarizer Agent")
st.subheader("Powered by Gemini 1.5 Flash + DuckDuckGo")

# --- API Key Validation ---
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in a .env file.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# --- DuckDuckGo Search Helper ---
def duckduckgo_search(query, max_results=5):
    with DDGS() as ddgs:
        return [r['body'] for r in ddgs.text(query, max_results=max_results)]

# --- File Upload Section ---
video_file = st.file_uploader(
    "Upload a video file",
    type=['mp4', 'mov', 'avi'],
    help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content...",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Uploading video and analyzing with AI..."):
                    # Upload video to Gemini
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Contextual Web Search
                    web_context = "\n".join(duckduckgo_search(user_query))

                    # Prompt Construction
                    analysis_prompt = f"""
Analyze the uploaded video for content and context.
Respond to the following query using video insights and supplementary web research.

Query: {user_query}

Web context:
{web_context}

Provide a detailed, user-friendly, and actionable response.
"""

                    # Generate response
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(
    [analysis_prompt, processed_video],
    generation_config={
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
)

                    

                # Display results
                st.subheader("📊 Analysis Result")
                st.markdown(response.text)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video file to begin analysis.")

st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(to right, pink, #e0eafc, green);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Stylish Text Area */
    .stTextArea textarea {
        height: 70px !important;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        background: linear-gradient(to right, pink, #e0eafc, #cfdef3);
        border: 1px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(to right, #e0eafc, #cfdef3);
        color: black;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        border: none;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s ease;
        box-shadow: 0 4px 8px rgba(79, 139, 249, 0.3);
    }

    .stButton>button:hover {
        background:white;
        box-shadow: 0 6px 14px rgba(58, 111, 227, 0.4);
    }

    /* Container Styling (optional glassmorphism) */
    .block-container {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }

    h1, h2, h3 {
        color: #343a40;
    }

    .css-1cpxqw2 edgvbvh3 {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)

