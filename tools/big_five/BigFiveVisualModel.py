import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.abspath(os.path.join("3DDFA_V2")))
from myutils.model_function import (
    model_from_json,
    format_3d_landmarks_by_frame,
    compute_data_features,
    compute_frame_difference,
)
from myutils.landmarks_processing import (
    translate_face_to_origin,
    rotate_face_to_match_z_axis,
    rotate_face_to_parallel_x_axis,
)
from myutils.detect3d import detect_3d_landmarks, detect_3d_with_quality_assessment_2

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


class BigFiveVisualModel(object):
    def __init__(
        self,
        model_weights_path,
        model_architect_path,
        input_scaler_path,
        scale_min=0,
        scale_max=40,
        dataset_min=-4,
        dataset_max=4,
    ):
        with open(model_architect_path, "r") as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(model_weights_path)
        min_max_array = np.array([dataset_min, dataset_max]).reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(scale_min, scale_max)).fit(
            min_max_array
        )
        self.input_scaler = joblib.load(input_scaler_path)

    def predict_bigfive(
        self,
        video_path,
        frame_num_to_extract: int = 500,
        start_from_frame: int = 90,
        config: str = "configs/mb1_120x120.yml",
        mode: str = "gpu",
        opt: str = "2d_sparse",
        onnx: bool = True,
        export_video_result: bool = False,
    ):
        """
        Predict Big Five using with video
        Parameters:
        ------------------------
        video_path: Path to video
        frame_num_to_extract: Number of frames to extract
        start_from_frame: Skip some frame at start video
        mode: CPU or GPU, default is CPU
        export_video_result: If this is true, video result of 3d landmarks detection will be saved in examples/results folder
        ------------------------
        """
        print("Processing video: ", video_path)
        print("Detecting 3d landmarks")
        try:
            # if True:
            # Detect 3d landmarks on face
            landmarks_data_list = detect_3d_landmarks(
                video_fp=video_path,
                frame_num_to_extract=frame_num_to_extract,
                start_from_frame=start_from_frame,
                config=config,
                mode=mode,
                opt=opt,
                onnx=onnx,
                export_video_result=export_video_result,
            )

            if landmarks_data_list == None:
                return None

            print("Standardlizing 3d landmarks")
            # Standardlize 3d landmarks
            face_data_list = []
            for i in range(len(landmarks_data_list)):
                landmarks_data = landmarks_data_list[i]
                face_landmarks = format_3d_landmarks_by_frame(landmarks_data)
                face_landmarks = translate_face_to_origin(
                    anchor_point=29, face_matrix=face_landmarks
                )
                face_landmarks = rotate_face_to_match_z_axis(
                    sub_anchor_point=28, face_matrix=face_landmarks
                )
                face_landmarks = rotate_face_to_parallel_x_axis(
                    left_anchor_point=40,
                    right_anchor_point=43,
                    face_matrix=face_landmarks,
                )
                face_data_list.append(face_landmarks.values[:, 1:])
            face_data_list = np.array(face_data_list)

            print("Computing features")
            # Compute features
            X_data = compute_frame_difference(face_data_list)
            X_data = compute_data_features(X_data)
            # print(X_data)

            # Scale input data
            X_data = self.input_scaler.transform(X_data.reshape(1, X_data.shape[0]))
            X_data = X_data.reshape(1, 24, 17, 1)

            # Prediction
            y_pred = self.model.predict(X_data)
            y_pred = y_pred[0]
            print("Raw output:", y_pred)

            # Scale score, default scale range is 0->40
            y_pred = self.scale_score(y_pred)

            pred_dict = {
                "o": y_pred[0],
                "c": y_pred[1],
                "e": y_pred[2],
                "a": y_pred[3],
                "n": y_pred[4],
            }

            return pred_dict
        except Exception as e:
            print("ERROR", e)
            return None

    def scale_score(self, score_list) -> np.array:
        score_list = np.array(score_list).reshape(-1, 1)
        scaled_scores = self.scaler.transform(score_list).flatten()
        return scaled_scores

    def inverse_scaled_score(self, score_list) -> np.array:
        score_list = np.array(score_list).reshape(-1, 1)
        reversed_scores = self.scaler.inverse_transform(score_list).flatten()
        return reversed_scores
