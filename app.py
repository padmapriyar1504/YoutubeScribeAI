import os
import subprocess
import json
import requests
from flask import Flask, request, render_template,jsonify, render_template
from flask_socketio import SocketIO, emit
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
from keybert import KeyBERT
import openai


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your Vosk model directory
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# Replace with your Gemini API key and openai api key here


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









# def fetch_answer_from_gemini(question, context):
#     video_content = f"Context: {context}\nQuestion: {question}\nAnswer:"
#     headers = {"Content-Type": "application/json"}
#     payload = {"contents": [{"parts": [{"text": video_content}]}]}
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         response_data = response.json()
#         answer = response_data["candidates"][0]["content"]["parts"][0]["text"]
#         return answer
#     else:
#         return f"Error: {response.status_code}, {response.text}"



# @app.route("/chatbot", methods=["POST"])
# def chatbot():
#     data = request.json
#     question = data.get("question")
#     title = data.get("title", "")
#     description = data.get("description", "")
#     transcription = data.get("transcription", "")
    
#     context = f"{title}\n{description}\n{transcription}"
#     answer = fetch_answer_from_gemini(question, context)
#     return jsonify({"answer": answer})



OUT_OF_CONTEXT_PHRASES = [
    "answer not provided in text",
    "unable to find an answer",
    "not available in context",
    "question not related to the text",
    "no relevant answer",
        "the text does not provide this information",
    "this is not mentioned in the text",
    "cannot find this in the text",
    "no reference to this in the text",
    "this detail is not available in the text",
    "the text does not include this topic",
    "not covered in the text",
    "there is no mention of this in the text",
    "unable to locate this in the text",
    "this information is missing from the text",
        "cannot be answered from the given source",
    "not in the video",
    "does not contain information",
    "this question is outside the scope",
    "not mentioned",
    "no data provided",
    "not available in the context",
    "doesn't contain information",
    "does not contain information",
    "not in the text",
    "this question cannot be answered from the given text",
    "information not found",
    "cannot provide an answer based on the context",
    "irrelevant to the provided content",
    "no relevant information provided",
    "not addressed in the source",
    "no context available for this question",
    "not supported by the given data",
    "the context does not address this",
    "outside the context of the source",
    "insufficient information to answer",
    "context does not include details on this",
    "no evidence in the text to support this",
    "the source does not cover this topic",
    "unable to answer with the provided content",
    "this falls outside the provided context",
    "no mention of this in the text",
    "does not align with the context provided",
    "the input does not include this information",
    "cannot deduce an answer from the given data",
    "I cannot answer",
    "unknown",
    "not known",
    "the answer cannot",
    "cannot be found in the given context",
    "doesn't mention",
    "does not mention",
    "There is no mention",
    "not provided",
    "doesn't describe",
    "does not describe"
]

# def fetch_answer_from_gemini(question, context, general=False):
#     if general:
#         # Use a fallback model for general chatbot queries
#         model_type = "chat-bison-001"  # Replace with a valid general-purpose model from ListModels
#     else:
#         model_type = "gemini-1.5-flash"  # Use this for video-context queries
    
#     video_content = f"Context: {context}\nQuestion: {question}\nAnswer:" if not general else f"Question: {question}\nAnswer:"
#     headers = {"Content-Type": "application/json"}
#     payload = {"contents": [{"parts": [{"text": video_content}]}]}
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_type}:generateContent?key={GEMINI_API_KEY}"

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         response_data = response.json()
#         try:
#             answer = response_data["candidates"][0]["content"]["parts"][0]["text"]
#             return answer
#         except (KeyError, IndexError):
#             return "Unable to parse the response from Gemini API."
#     elif response.status_code == 404:
#         return "Model not found or unsupported. Please check the API and model configuration."
#     else:
#         return f"Error: {response.status_code}, {response.text}"



# @app.route("/chatbot", methods=["POST"])
# def chatbot():
#     data = request.json
#     question = data.get("question")
#     title = data.get("title", "")
#     description = data.get("description", "")
#     transcription = data.get("transcription", "")
    
#     context = f"{title}\n{description}\n{transcription}"
#     answer = fetch_answer_from_gemini(question, context)
    
#     # Check if the answer is out of context or an error response
#     if any(phrase in answer.lower() for phrase in OUT_OF_CONTEXT_PHRASES) or "error:" in answer.lower():
#         # Switch to general chatbot
#         answer = fetch_answer_from_gemini(question, context="", general=True)
#         source = "general chatbot"
#     else:
#         source = "video context"
    
#     return jsonify({"answer": answer, "source": source})

def fetch_answer_from_gemini(question, context, general=False):
    """
    Fetches an answer from Gemini API using video context or general fallback.
    """
    model_type = "chat-bison-001" if general else "gemini-1.5-flash"
    video_content = (
        f"Context: {context}\nQuestion: {question}\nAnswer:"
        if not general
        else f"Question: {question}\nAnswer:"
    )
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": video_content}]}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_type}:generateContent?key={GEMINI_API_KEY}"

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        try:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Unable to parse the response from Gemini API."
    elif response.status_code == 404:
        return "Model not found or unsupported. Please check the API and model configuration."
    else:
        return f"Error: {response.status_code}, {response.text}"

def gpt_chatbot(question):
    """
    Handles out-of-context questions using GPT-4 or GPT-3.5 API.
    """
    prompt = f"""
    You are a knowledgeable assistant with access to web search capabilities. A user has asked the following question:

    "{question}"

    Please provide a brief and accurate answer as if you searched the web. Include relevant sources to back up your response. Clearly list these sources at the end.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and factual assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response["choices"][0]["message"]["content"]

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Chatbot endpoint to handle user queries.
    """
    data = request.json
    question = data.get("question")
    title = data.get("title", "")
    description = data.get("description", "")
    transcription = data.get("transcription", "")

    # Construct context
    context = f"{title}\n{description}\n{transcription}"
    answer = fetch_answer_from_gemini(question, context)

    if any(phrase in answer.lower() for phrase in OUT_OF_CONTEXT_PHRASES) or "error:" in answer.lower():
        # Switch to GPT chatbot for out-of-context questions
        answer = gpt_chatbot(question)
        source = "GPT-3.5/4 with web search simulation"
    else:
        source = "video context"

    return jsonify({"answer": answer, "source": source})

if __name__ == "__main__":
    socketio.run(app, debug=True)