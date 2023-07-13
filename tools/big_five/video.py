import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from call_funtions import load_model, predict_bigfive


def init_video():
    return load_model()


def handle_video(session_id: str, model):
    video = f"../../public/interview/{session_id}/main.mp4"
    bigfive_result = predict_bigfive(
        model=model,
        video_path=video,
        frame_num_to_extract=700,
        start_from_frame=5,
        mode="cpu",
    )
    return bigfive_result


app = Flask(__name__)
CORS(app)
app.static_folder = "public"


video_model = init_video()


@app.route("/get_video", methods=["GET"])
async def big_five_report():
    session_id: int = request.args.get("session_id")
    result = handle_video(session_id, video_model)
    data = {key: int(value) for key, value in result.items()}
    return jsonify(data), 200


if __name__ == "__main__":
    app.run()
