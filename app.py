import os
import subprocess
import json
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
import requests

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your Vosk model directory
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# Replace with your actual Gemini API key
GEMINI_API_KEY = 'AIzaSyBGRyPEB-v-cnhAgVhDXUd_6pcwyou7oFU'  # Replace with your actual API key


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
    Transcribes audio using the Vosk model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please download it first.")

    processed_audio_file = os.path.join(UPLOAD_FOLDER, "processed_audio.wav")
    preprocess_audio(audio_file, processed_audio_file)

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    with wave.open(processed_audio_file, "rb") as wf:
        transcription = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcription.append(result.get("text", ""))

        final_result = json.loads(recognizer.FinalResult())
        transcription.append(final_result.get("text", ""))

    return " ".join(transcription)


def summarize_text(transcription_text):
    """
    Summarizes the transcription text using Gemini API.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": transcription_text}]
        }]
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        summary = response_data['candidates'][0]['content']['parts'][0]['text']
        return summary
    else:
        raise RuntimeError(f"Error summarizing text: {response.status_code}, {response.text}")


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

        # Step 1: Extract video info
        info_command = ["yt-dlp", "--dump-json", url]
        result = subprocess.run(info_command, stdout=subprocess.PIPE, check=True, text=True)
        video_info = json.loads(result.stdout)

        title = video_info.get("title", "unknown_title").replace(" ", "_")
        description = video_info.get("description", "No description available.")
        emit("update", {"step": "info", "title": title, "description": description})

        # Step 2: Download audio
        wav_file = os.path.join(UPLOAD_FOLDER, f"{title}.wav")
        audio_command = [
            "yt-dlp", "-f", "bestaudio", "--extract-audio", "--audio-format", "wav",
            "-o", wav_file, url
        ]
        subprocess.run(audio_command, check=True)

        # Step 3: Transcription
        transcription = transcribe_audio(wav_file, VOSK_MODEL_PATH)
        emit("update", {"step": "transcription", "transcription": transcription})

        # Step 4: Summarization
        summary = summarize_text(transcription)
        emit("update", {"step": "summary", "summary": summary})
    except subprocess.CalledProcessError as e:
        emit("error", {"message": f"Error downloading audio: {e}"})
    except json.JSONDecodeError:
        emit("error", {"message": "Failed to parse video information."})
    except Exception as e:
        emit("error", {"message": str(e)})


if __name__ == "__main__":
    socketio.run(app, debug=True)