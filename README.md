# AI Video Summarizer 🎥
AI Video Summarizer is a web-based application that leverages Google's Gemini AI to analyze and summarize video content. Built with Streamlit, this tool offers an intuitive interface for uploading videos and receiving concise summarie.

## Features

- **Video Upload** Easily upload video files for analysi.
- **AI-Powered Summarization** Utilizes Google's Gemini AI to generate summarie.
- **User-Friendly Interface** Built with Streamlit for a seamless user experienc.
- **Real-Time Feedback** Displays analysis results promptly after processin.

## Getting Started

### Prerequisites

 Python 3.8 or highr
 Google Generative AI SDK (`google-generativeai)
 Streamlt

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/git-devisha/AI-Video-Summarizer.git
   cd AI-Video-Summarizer
   ``
2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ``
3. **Set up environment variables**:

   Create a `.env` file in the root directory and add your Google Generative AI API key:

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ``
## Usage
Run the Streamlit applicatio:
```bash
streamlit run app1.py
``
This will open the application in your default web browser. Upload a video file, and the application will process and display the summarized conten.

## Project Structure
```
AI-Video-Summarizer/
├── app.py             # Main application file
├── app1.py            # Alternative application file
├── requirements.txt   # List of dependencies
├── .env               # Environment variables
└── README.md          # Project documentation
``
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixe.
