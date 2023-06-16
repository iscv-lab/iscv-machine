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
  landmarks_features_list = []
  for i in range(0, 68):
    for d in range(0, 3):
      landmark_std = np.std(data_array[:, i, d])
      landmarks_features_list.append(landmark_std)

      landmark_mean = np.mean(data_array[:, i, d])
      landmarks_features_list.append(landmark_mean)

  return np.array(landmarks_features_list)

def compute_frame_difference(data_array):
  nb_frames = data_array.shape[0]
  new_obs = []
  for frame_index in range(1, nb_frames):
    new_f = []
    for p in range(0, 68):
      new_p = []
      for d in range(0, 3):
        new_d = data_array[frame_index, p, d] - data_array[frame_index - 1, p, d]
        new_p.append(new_d)
      new_f.append(new_p)
    new_obs.append(new_f)
  return np.array(new_obs)
