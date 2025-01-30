# YoutubeScribeAI

In an era dominated by video content, extracting key insights from lengthy YouTube videos can be challenging. This project introduces a streamlined web application that automates video transcription, summarization, timeline generation, and chatbot-based interaction to enhance user experience.

Key Features

YouTube URL Input: Users can submit a video link for processing.

Audio Extraction & Transcription: Uses yt-dlp and FFmpeg to convert audio into .wav, transcribed by the Vosk model.

Summarization: The Gemini model generates concise video summaries.

Timeline Generation: KeyBERT extracts key points per minute for structured insights.

Interactive Chatbot: A Gemini-powered chatbot answers video-specific questions and switches to general mode for broader queries.

User-Friendly Interface: Built with HTML, CSS, and Flask for responsiveness.

Deployment & Testing: Docker ensures smooth deployment, while Selenium optimizes the user experience.

Version Control: Managed with Git for efficient collaboration.

This application enables users to save time and gain insights effortlessly from YouTube videos. 🚀
