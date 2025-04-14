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

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")

# --- API Key Check ---
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in a .env file.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# --- Helper Function: DuckDuckGo Search ---
def duckduckgo_search(query, max_results=5):
    return [f"Simulated search result: '{query}'"] * max_results


# def duckduckgo_search(query, max_results=5):
#     with DDGS() as ddgs:
#         return [r['body'] for r in ddgs.text(query, max_results=max_results)]

# --- File Upload ---
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
                    # Upload video
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Contextual Search
                    web_context = "\n".join(duckduckgo_search(user_query))

                    # Prompt for analysis
                    analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using video insights and supplementary web research:

                    Query: {user_query}

                    Web context:
                    {web_context}

                    Provide a detailed, user-friendly, and actionable response.
                    """

                    # Generate response from Gemini
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(
                        analysis_prompt,
                        generation_config={
                            "temperature": 0.7,
                            "top_p": 1,
                            "top_k": 1,
                            "max_output_tokens": 2048,
                        }
                    )

                st.subheader("Analysis Result")
                st.markdown(response.text)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

# --- UI Customization ---
st.markdown("""
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
""", unsafe_allow_html=True)
