import os
from google.cloud import texttospeech
import os
from google.cloud import speech_v1p1beta1 as speech


def speech_to_text(path: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_PATH")

    def transcribe_speech(audio_file):
        client = speech.SpeechClient()

        # Load the audio file
        with open(audio_file, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,  # Update with the correct sample rate of your audio
            language_code="vi-VN",  # Update with the correct language code for Vietnamese
        )

        response = client.recognize(config=config, audio=audio)

        # Extract and return the recognized text
        if len(response.results) > 0:
            return response.results[0].alternatives[0].transcript
        else:
            return ""

    # Specify the path to your output.mp3 file

    # Transcribe the speech to text
    transcribed_text = transcribe_speech(path)

    return transcribed_text
