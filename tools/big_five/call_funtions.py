from BigFiveVisualModel import BigFiveVisualModel


def load_model(
    model_weights_path="saved_model/CNN_LSTM_model_weights.h5",
    model_architect_path="saved_model/CNN_LSTM_model_architect.json",
    input_scaler_path="saved_model/my_scaler.pkl",
    scale_min=0,
    scale_max=40,
):
    model = BigFiveVisualModel(
        model_weights_path=model_weights_path,
        model_architect_path=model_architect_path,
        input_scaler_path=input_scaler_path,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    print("Loaded model!")
    return model


def predict_bigfive(
    model,
    video_path: str,
    frame_num_to_extract: int = 700,
    start_from_frame: int = 100,
    mode: str = "cpu",
):
    if model is None:
        raise TypeError("Model is not loaded!")

    if not video_path:
        raise TypeError("Invalid video path!")

    onnx = mode == "cpu"
    bigfive_result = model.predict_bigfive(
        video_path=video_path,
        frame_num_to_extract=frame_num_to_extract,
        start_from_frame=start_from_frame,
        mode=mode,
        onnx=onnx,
    )
    return bigfive_result
