import os
import zipfile
import requests

# Define the URL for the Vosk model
vosk_model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
vosk_model_path = "vosk-model-small-en-us"

# Download the model if it doesn't exist
if not os.path.exists(vosk_model_path):
    print("Downloading Vosk model...")
    model_zip_path = "vosk-model-small-en-us.zip"
    response = requests.get(vosk_model_url, stream=True)

    # Save the zip file
    with open(model_zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")

    # Extract the zip file
    print("Extracting the Vosk model...")
    with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")  # Extract to current directory
    print(f"Model extracted to {vosk_model_path}.")

    # Clean up the zip file
    os.remove(model_zip_path)
else:
    print(f"Vosk model already exists at {vosk_model_path}.")