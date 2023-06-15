import sys, os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
sys.path.append(os.path.abspath(os.path.join('3DDFA_V2')))

from tensorflow.keras.models import load_model
from myutils.model_function import format_3d_landmarks_by_frame, compute_data_features
from myutils.landmarks_processing import translate_face_to_origin, rotate_face_to_match_z_axis, rotate_face_to_parallel_x_axis
from myutils.detect3d import detect_3d_with_quality_assessment

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BigFiveVisualModel(object):

    def __init__(self, model_path, scale_min = 0, scale_max = 40, dataset_min = -5, dataset_max = 5):
        self.model = load_model(model_path)
        min_max_array = np.array([dataset_min, dataset_max]).reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(scale_min, scale_max)).fit(min_max_array)

    def predict_bigfive(self, video_path,
                        max_frame: int = 500, 
                        max_error: int = 100, 
                        min_bright_percent: float = 0.37, 
                        contrast_thresh: float = 0.3, 
                        blurry_thresh: int = 100, 
                        skip_from_frame: int = 90, 
                        config: str = 'configs/mb1_120x120.yml', 
                        mode: str='cpu', 
                        opt: str = '2d_sparse', 
                        onnx: bool = True, 
                        export_video_result: bool=False):
        """
        Predict Big Five using with video
        Parameters:
        ------------------------
        video_path: Path to video
        max_frame: Number of frames to extract
        max_error: Max error count for video. If error count > max_error, process will be stopped
        min_bright_percent: Threshold to determine if face image is bright or dark
        contrast_thresh: Threshold to determine if frame is low contrast or not
        blurry_thresh: Threshold to determine if frame is blurry or not
        skip_from_frame: Skip some frame at start video
        mode: CPU or GPU, default is CPU
        export_video_result: If this is true, video result of 3d landmarks detection will be saved in examples/results folder
        ------------------------
        """
        print("Processing video: ", video_path)
        print("Detecting 3d landmarks")
        try:
            # Detect 3d landmarks on face
            landmarks_data_list = detect_3d_with_quality_assessment(video_fp=video_path,
                                                                    max_frame=max_frame, 
                                                                    max_error=max_error, 
                                                                    min_bright_percent=min_bright_percent, 
                                                                    contrast_thresh=contrast_thresh, 
                                                                    blurry_thresh=blurry_thresh, 
                                                                    skip_from_frame=skip_from_frame, 
                                                                    config=config, 
                                                                    mode=mode, 
                                                                    opt=opt, 
                                                                    onnx=onnx, 
                                                                    export_video_result=export_video_result)

            if landmarks_data_list == None:
                return None

            print("Standardlizing 3d landmarks")
            # Standardlize 3d landmarks
            face_data_list = []
            for i in range(len(landmarks_data_list)):
                landmarks_data = landmarks_data_list[i]
                face_landmarks = format_3d_landmarks_by_frame(landmarks_data)
                face_landmarks = translate_face_to_origin(anchor_point=29, face_matrix=face_landmarks)
                face_landmarks = rotate_face_to_match_z_axis(sub_anchor_point=28, face_matrix=face_landmarks)
                face_landmarks = rotate_face_to_parallel_x_axis(left_anchor_point=40, right_anchor_point=43, face_matrix=face_landmarks)
                face_data_list.append(face_landmarks.values[:, 1:])
            face_data_list = np.array(face_data_list)

            print("Computing features")
            # Compute features
            X_data = compute_data_features(face_data_list)
            X_data = X_data.reshape(1, 24, 17, 1)

            # Prediction
            y_pred = self.model.predict(X_data)
            y_pred = y_pred[0]

            # Scale score, default scale range is 0->40
            y_pred = self.scale_score(y_pred)

            return y_pred
        except Exception as e:
            print('ERROR', e)
            return None

    def scale_score(self, score_list) -> np.array:
        score_list = np.array(score_list).reshape(-1, 1)
        scaled_scores = self.scaler.transform(score_list).flatten()
        return scaled_scores

    def inverse_scaled_score(self, score_list) -> np.array:
        score_list = np.array(score_list).reshape(-1, 1)
        reversed_scores = self.scaler.inverse_transform(score_list).flatten()
        return reversed_scores