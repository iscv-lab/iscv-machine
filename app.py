from flask import Flask, request, jsonify, send_file
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pydub import AudioSegment
import requests
import os
from dotenv import load_dotenv
import glob
from flask_cors import CORS

load_dotenv()  # Load environment variables from .env file

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

app = Flask(__name__)
CORS(app)
app.static_folder = "public"


@app.route("/data", methods=["POST"])
def post_data():
    data = request.get_json()  # Get the JSON data from the request

    # Process the data here
    # ...

    request_data = request.get_json()
    # Extract the string "text" from the request body
    text = request_data.get("text")
    print(text)
    response = {"message": "Data received successfully"}
    return jsonify(response), 200


@app.route("/data", methods=["GET"])
def get_data():
    # Retrieve data from the server or perform any other operations
    data = {"example_key": "example_value"}

    return jsonify(data), 200


@app.route("/text2speech", methods=["POST"])
def text_to_speech():
    body = request.get_json()
    # Retrieve data from the server or perform any other operations
    import os
    from google.cloud import texttospeech

    # Đường dẫn đến tệp keyfile.json
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = "/Users/khanhle/Documents/work/iscv/interview/iscv.json"

    def text_to_speech(text, output_file):
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="vi-VN", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
            print(f"Đã lưu file âm thanh: {output_file}")

    text = body.get("text")
    output_file = f"./public/translate/{uuid.uuid4()}.mp3"
    Path("./public/translate/").mkdir(parents=True, exist_ok=True)
    text_to_speech(text, output_file)

    response = send_file(output_file, mimetype="audio/mpeg"), 200

    # os.remove(output_file)

    return response


@app.route("/generate_text", methods=["POST"])
def generate_text():
    from utils.translate import speech_to_text

    interviewid = request.args.get("interviewid")
    wav_url = f"{os.getenv('NODEJS_ENDPOINT')}public/interview/{interviewid}/audio.wav"
    response = requests.get(wav_url)
    wav_filename = "audio.wav"
    mp3_filename = "audio.mp3"

    with open(wav_filename, "wb") as wav_file:
        wav_file.write(response.content)

    audio = AudioSegment.from_wav(wav_filename)
    audio.export(mp3_filename, format="mp3")

    destination_folder = f"./public/interview/{interviewid}"
    isExist = os.path.exists(destination_folder)
    if not isExist:
        os.makedirs(destination_folder)

    mp3_path = os.path.join(destination_folder, mp3_filename)
    os.rename(mp3_filename, mp3_path)

    os.remove(wav_filename)

    text = speech_to_text(mp3_path)

    return jsonify(text), 200


@app.route("/big_five", methods=["GET"])
def big_five():
    from utils.video import (
        download_webm,
        convert_webm_to_mp4,
        convert_webm_to_mp3,
        split_video,
    )

    from tools.big_five.index import handle_big_five
    from utils.text import download_txt
    from tools.big_five.report import handle_report

    employeeid = request.args.get("employee_id")
    employeename = request.args.get("employee_name")
    interviewid = request.args.get("interview_id")
    video_url = (
        f"{os.getenv('NODEJS_ENDPOINT')}public/interview/{interviewid}/video.webm"
    )
    qa_url = f"{os.getenv('NODEJS_ENDPOINT')}public/interview/{interviewid}/qa.txt"

    destPath = f"./public/interview/{interviewid}/"

    # Check if the folder already exists

    if not os.path.exists(destPath):
        # Create the folder
        os.makedirs(destPath)
        print("Folder created successfully.")
    else:
        file_paths = glob.glob(os.path.join(destPath, "*"))

        # Iterate through each file and remove it
        for file_path in file_paths:
            os.remove(file_path)
        print("Folder already exists.")

    executor = ThreadPoolExecutor()
    # Download the WebM file
    webm_task = executor.submit(
        partial(download_webm, video_url, destPath + "video.webm")
    )
    qa_task = executor.submit(partial(download_txt, qa_url, destPath, "qa.txt"))
    webm_task.result()
    qa_task.result()
    # Create a thread pool executor

    # Submit tasks for conversion of WebM to MP4 and MP3
    mp4_task = executor.submit(
        partial(convert_webm_to_mp4, destPath + "video.webm", destPath + "video.mp4")
    )
    mp3_task = executor.submit(
        partial(convert_webm_to_mp3, destPath + "video.webm", destPath + "audio.mp3")
    )

    # Wait for the tasks to complete
    mp4_task.result()
    mp3_task.result()

    # Split the video
    split_video(
        destPath + "video.mp4", destPath + "introduction.mp4", destPath + "main.mp4", 90
    )

    result = handle_big_five(interviewid)
    handle_report(employeeid, employeename, interviewid)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run()
