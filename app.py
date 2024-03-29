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
import asyncio
import subprocess
import threading
import logging
from utils.file import remove_files_async
from utils.log import thread_log
from tools.job.recommend import load_recommend_model


load_dotenv()  # Load environment variables from .env file

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
app = Flask(__name__)
CORS(app)
app.static_folder = "public"

data_recommend, model_recommend = load_recommend_model()


@app.route("/data", methods=["POST"])
def post_data():
    data = request.get_json()  # Get the JSON data from the request

    # Process the data here
    # ...

    request_data = request.get_json()
    # Extract the string "text" from the request body
    text = request_data.get("text")

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


@app.route("/big_five/start", methods=["GET"])
def big_five_start():
    def run_subprocess(session_id):
        thread_log()
        # load_dotenv()
        from utils.video import (
            download_webm,
            convert_webm_to_mp4,
            convert_webm_to_mp3,
            split_video,
        )

        from utils.text import download_txt

        try:
            video_url = f"{os.getenv('NODEJS_ENDPOINT')}public/interview/{session_id}/video.webm"
            qa_url = (
                f"{os.getenv('NODEJS_ENDPOINT')}public/interview/{session_id}/qa.txt"
            )

            destPath = f"./public/interview/{session_id}/"

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
                partial(
                    convert_webm_to_mp4, destPath + "video.webm", destPath + "video.mp4"
                )
            )
            # mp3_task = executor.submit(
            #     partial(
            #         convert_webm_to_mp3, destPath + "video.webm", destPath + "audio.mp3"
            #     )
            # )

            # Wait for the tasks to complete
            mp4_task.result()
            # mp3_task.result()

            # Split the video
            split_video(
                destPath + "video.mp4",
                destPath + "introduction.mp4",
                destPath + "main.mp4",
                15,
            )
            print(
                f"{os.getenv('NODEJS_ENDPOINT')}python/big_five/started?session_id={session_id}"
            )
            response = requests.get(
                f"{os.getenv('NODEJS_ENDPOINT')}python/big_five/started?session_id={session_id}"
            )
            if response.status_code == 200:
                print("Response:", response.json())
            else:
                print(
                    "start POST request failed with status code:", response.status_code
                )
        except Exception as e:
            print(e)

    session_id = request.args.get("session_id")
    subprocess_thread = threading.Thread(target=run_subprocess, args=(session_id,))
    subprocess_thread.start()
    return jsonify("approved"), 200


@app.route("/big_five/audio", methods=["GET"])
def big_five_audio():
    from tools.big_five.audio import handle_big_five_audio

    def run_subprocess(session_id):
        thread_log()
        try:
            result = handle_big_five_audio(session_id)
            response = requests.post(
                f"{os.getenv('NODEJS_ENDPOINT')}python/big_five/audio?session_id={session_id}",
                json=result,
            )
            if response.status_code != 200:
                print(
                    "audio POST request failed with status code:", response.status_code
                )
        except Exception as e:
            print(e)

    session_id = request.args.get("session_id")
    subprocess_thread = threading.Thread(target=run_subprocess, args=(session_id,))
    subprocess_thread.start()
    return jsonify("audio_approved"), 200


@app.route("/big_five/video", methods=["GET"])
def big_five_video():
    session_id: int = request.args.get("session_id")

    def run_subprocess(session_id):
        # Configure logging for the subprocess
        thread_log()
        try:
            response = requests.get(
                f"{os.getenv('VIDEO_ENDPOINT')}get_video?session_id={session_id}"
            )
            if response.status_code == 200:
                result = response.json()
                callback = requests.post(
                    f"{os.getenv('NODEJS_ENDPOINT')}python/big_five/video?session_id={session_id}",
                    json=result,
                )
                if callback.status_code != 200:
                    print(
                        "video POST request failed with status code:",
                        response.status_code,
                    )
            else:
                print(
                    "mini POST request failed with status code:", response.status_code
                )
        except Exception as e:
            print(e)
        # Run the subprocess in a separate thread

    subprocess_thread = threading.Thread(target=run_subprocess, args=(session_id,))
    subprocess_thread.start()
    return jsonify("video_approved"), 200


@app.route("/big_five/report", methods=["POST"])
async def big_five_report():
    from tools.big_five.report import handle_report

    print("start report")
    data = request.get_json()
    await handle_report(data)
    return jsonify("success"), 200


@app.route("/big_five/clean", methods=["GET"])
async def big_five_clean():
    session_id: int = request.args.get("session_id")
    folder_path = f"{ROOT_DIR}/public/interview/{session_id}/"
    file_paths = [
        folder_path + "chat.png",
        folder_path + "introductrion.mp4",
        folder_path + "main.mp4",
        folder_path + "qa.txt",
        folder_path + "report.docx",
        folder_path + "video.mp4",
        folder_path + "video.webm",
    ]

    await remove_files_async(file_paths)

    return jsonify("success"), 200


@app.route("/job/recommend", methods=["POST"])
async def job_recommend():
    from tools.job.recommend import recommended_jobs

    data = request.get_json()
    result = recommended_jobs(data["skills"], data_recommend, model_recommend)
    return jsonify(result), 200


@app.route("/test", methods=["GET"])
def test():
    return jsonify("hello"), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4002, debug=True)
