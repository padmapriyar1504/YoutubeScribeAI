import os
import subprocess
import json
import requests
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
from keybert import KeyBERT

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your Vosk model directory
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# Replace with your Gemini API key
GEMINI_API_KEY = ""

# Initialize KeyBERT model
kw_model = KeyBERT()


def preprocess_audio(audio_file, output_file):
    """
    Converts the given audio file to mono and 16kHz.
    """
    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        raise RuntimeError(f"Error during audio preprocessing: {e}")


def transcribe_audio(audio_file, model_path):
    """
    Transcribes audio using the Vosk model and segments the transcription by minute.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please download it first.")

    processed_audio_file = os.path.join(UPLOAD_FOLDER, "processed_audio.wav")
    preprocess_audio(audio_file, processed_audio_file)

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    with wave.open(processed_audio_file, "rb") as wf:
        transcription_by_minute = []
        frame_rate = wf.getframerate()

        current_minute = 0
        transcription = ""

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                transcription += text + " "

                elapsed_time = wf.tell() / frame_rate
                if elapsed_time // 60 > current_minute:
                    transcription_by_minute.append({
                        "time": f"{current_minute}-{current_minute + 1} min",
                        "text": transcription.strip()
                    })
                    transcription = ""
                    current_minute += 1

        final_result = json.loads(recognizer.FinalResult())
        transcription += final_result.get("text", "")
        transcription_by_minute.append({
            "time": f"{current_minute}-{current_minute + 1} min",
            "text": transcription.strip()
        })

    return transcription_by_minute


def extract_top_keyword(text):
    """
    Extracts the top keyword for a given text using KeyBERT.
    """
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return keywords[0][0] if keywords else "No key topic found"


def extract_key_topics(transcription_by_minute):
    """
    Extracts key topics for each minute's transcription and formats them as a timeline.
    """
    timeline = []
    for segment in transcription_by_minute:
        top_keyword = extract_top_keyword(segment["text"])
        timeline.append(f"{segment['time']}: {top_keyword}")
    return timeline


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("process_url")
def process_url(data):
    try:
        url = data.get("url")
        if not url:
            emit("error", {"message": "No URL provided."})
            return

        # Extract video info
        info_command = ["yt-dlp", "--dump-json", url]
        result = subprocess.run(info_command, stdout=subprocess.PIPE, check=True, text=True)
        video_info = json.loads(result.stdout)

        title = video_info.get("title", "unknown_title").replace(" ", "_")
        description = video_info.get("description", "No description available.")
        emit("update", {"step": "info", "title": title, "description": description})

        # Download audio
        wav_file = os.path.join(UPLOAD_FOLDER, f"{title}.wav")
        audio_command = [
            "yt-dlp", "-f", "bestaudio", "--extract-audio", "--audio-format", "wav",
            "-o", wav_file, url
        ]
        subprocess.run(audio_command, check=True)

        # Transcription
        transcription_by_minute = transcribe_audio(wav_file, VOSK_MODEL_PATH)
        emit("update", {"step": "transcription", "transcription": transcription_by_minute})

        # Key Topic Extraction
        timeline = extract_key_topics(transcription_by_minute)
        emit("update", {"step": "topics", "key_topics": timeline})

        # Summarization
        full_transcription = " ".join([segment["text"] for segment in transcription_by_minute])
        summary = summarize_text(full_transcription)
        emit("update", {"step": "summary", "summary": summary})

    except subprocess.CalledProcessError as e:
        emit("error", {"message": f"Error downloading audio: {e}"})
    except json.JSONDecodeError:
        emit("error", {"message": "Failed to parse video information."})
    except Exception as e:
        emit("error", {"message": str(e)})


def summarize_text(transcription_text):
    """
    Summarizes the transcription text using Gemini API.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": transcription_text}]}]
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        summary = response_data['candidates'][0]['content']['parts'][0]['text']
        return summary
    else:
        raise RuntimeError(f"Error summarizing text: {response.status_code}, {response.text}")


if __name__ == "__main__":
    socketio.run(app, debug=True)
