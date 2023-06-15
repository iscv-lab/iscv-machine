from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd
from typing import List, Tuple

def format_shape(face_landmarks):
  data = []
  for i in range(len(face_landmarks[0])):
    data.append((face_landmarks[0][i], face_landmarks[1][i], face_landmarks[2][i]))
  return data

def format_3d_landmarks_by_frame(face_landmarks):
  face_landmarks = format_shape(face_landmarks)
  face_landmarks = np.array(face_landmarks)
  point_names = np.arange(1, face_landmarks.shape[0] + 1, 1).reshape(face_landmarks.shape[0], 1)

  face_landmarks = np.hstack((point_names, face_landmarks))

  face_landmarks = pd.DataFrame(face_landmarks, columns=['point', 'x', 'y', 'z'])
  return face_landmarks

def reshape_data_to_expected_format(data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
  x_list = []
  y_list = []
  for i in range(0, len(data)):
    x_list.append(data[i][0])
    y_list.append(data[i][1])

  return (np.stack(x_list), np.stack(y_list))

def compute_data_features(data_array):
  row_data = []
  
  landmarks_features_list = []
  for i in range(0, 68):
    for d in range(0, 3):
      landmark_std = np.std(data_array[:, i, d])
      landmarks_features_list.append(landmark_std)
        
      landmark_mean = np.mean(data_array[:, i, d])
      landmarks_features_list.append(landmark_mean)
  row_data.append(landmarks_features_list)

  row_data = np.array(row_data)
  row_data = row_data.reshape(row_data.shape[0], row_data.shape[1], 1)
  return row_data
