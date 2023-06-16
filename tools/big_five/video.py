import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import argparse
from BigFiveVisualModel import BigFiveVisualModel
import json

# Create an argument parser
parser = argparse.ArgumentParser(description="Demo Big Five Script")

# Add the input parameters
parser.add_argument("--param1", type=int, help="Parameter 1 description")

# Parse the input arguments
args = parser.parse_args()

# Access the input parameters
session_id = args.param1

# Use the input parameters in your script
print("Parameter 1:", session_id)

from BigFiveVisualModel import BigFiveVisualModel

MIN_BRIGHT_PERCENT = 0.37
CONTRAST_THRESH = 0.3
BLURRY_THRESH = 100

model = BigFiveVisualModel(
    model_weights_path="saved_model/CNN_LSTM_model_weights.h5",
    model_architect_path="saved_model/CNN_LSTM_model_architect.json",
    input_scaler_path="saved_model/my_scaler.pkl",
    scale_min=0,
    scale_max=40,
)

videos = [f"../../public/interview/{session_id}/main.mp4"]

for vid in videos:
    # frame_num_to_extract=700 : lấy 700 frames
    # start_from_frame=100: skip 100 frames đầu tiên (dính đoạn mở đầu msteams)
    bigfive_result = model.predict_bigfive(
        video_path=vid, frame_num_to_extract=500, start_from_frame=0
    )

    print("------------------------------\nResult of video {0}:".format(vid))
    print("O-C-E-A-N:", bigfive_result)
    print("------------------------------\n")
    data = {
        "o": int(bigfive_result[0]),
        "c": int(bigfive_result[1]),
        "e": int(bigfive_result[2]),
        "a": int(bigfive_result[3]),
        "n": int(bigfive_result[4]),
    }

    # Save the data as JSON
    with open(
        f"../../public/interview/{session_id}/video.json", "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, allow_nan=False)
