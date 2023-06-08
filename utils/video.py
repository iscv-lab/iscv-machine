import subprocess
import requests
from pydub import AudioSegment
import os
from pydub import AudioSegment


def download_webm(url, path):
    response = requests.get(url)
    with open(path, "wb") as file:
        file.write(response.content)


def convert_webm_to_mp4(input_file, output_file):
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-preset",
        "medium",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_file,
    ]
    subprocess.call(cmd)


def convert_webm_to_mp3(input_file, output_file):
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-vn",  # Disable video stream
        "-c:a",
        "libmp3lame",  # Use MP3 audio codec
        "-q:a",
        "2",  # Set audio quality (0-9, 2 being good quality)
        output_file,
    ]
    subprocess.call(cmd)


def split_video(input_file, output_file_1, output_file_2, duration_1):
    cmd1 = [
        "ffmpeg",
        "-i",
        input_file,
        "-t",
        str(duration_1),
        "-c",
        "copy",
        output_file_1,
    ]
    subprocess.call(cmd1)

    cmd2 = [
        "ffmpeg",
        "-i",
        input_file,
        "-ss",
        str(duration_1),
        "-c",
        "copy",
        output_file_2,
    ]
    subprocess.call(cmd2)
