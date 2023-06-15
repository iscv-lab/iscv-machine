import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely import geometry

_pi_above_180 = math.pi/180
_180_above_pi = 180/math.pi

def get_3d_landmarks_by_frame(frame_index, anno, anno_keys):
  frame_key = anno_keys[frame_index]
  face_landmarks = list(anno[frame_key]['face']['landmarks'])

  face_landmarks = np.array(face_landmarks)
  point_names = np.arange(1, face_landmarks.shape[0] + 1, 1).reshape(face_landmarks.shape[0], 1)

  face_landmarks = np.hstack((point_names, face_landmarks))

  face_landmarks = pd.DataFrame(face_landmarks, columns=['point', 'x', 'y', 'z'])
  return face_landmarks

def rotate_around_axis(angle, point, axis='x'):
    global _pi_above_180
    radian = angle * _pi_above_180

    cos_rad = math.cos(radian)
    sin_rad = math.sin(radian)

    if axis == 'y':
        R_matrix = np.array([
            [cos_rad, 0, sin_rad],
            [0, 1, 0],
            [-sin_rad, 0, cos_rad]
        ])
    elif axis == 'z':
        R_matrix = np.array([
            [cos_rad, -sin_rad, 0],
            [sin_rad, cos_rad, 0],
            [0, 0, 1]
        ])
    else:
        R_matrix = np.array([
            [1, 0, 0],
            [0, cos_rad, -sin_rad],
            [0, sin_rad, cos_rad]
        ])

    result = np.matmul(R_matrix, point.reshape(3,))
    return result

def plot_3D_face(ax, face_matrix, c):
    x = face_matrix['x'].values
    y = face_matrix['y'].values
    z = face_matrix['z'].values
    color = ((face_matrix['point'] == 29) | (face_matrix['point'] == 28))
    color = color.map({False: c, True: 'r'})
    ax.scatter3D(xs=x, ys=y, zs=z, c=color)
    # ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False)

    point_28 = face_matrix[face_matrix['point'] == 28].values[0]
    point_29 = face_matrix[face_matrix['point'] == 29].values[0]
    ax.plot([point_28[1], point_29[1]], [point_28[2], point_29[2]], [point_28[3], point_29[3]], c='r', linewidth=1)

    # draw parallel line
    lap = face_matrix[face_matrix['point'] == 40].values[0]
    rap = face_matrix[face_matrix['point'] == 43].values[0]
    ax.plot([lap[1], rap[1]], [lap[2], rap[2]], [lap[3], rap[3]], c='r', linewidth=1)
    # midpoint = (lap + rap)/2
    # ax.plot([midpoint[1], 0], [midpoint[2], 0], [midpoint[3], midpoint[3]], c='r', linewidth=1)

    # draw x, y, z axis
    # axis_size = 120
    # line_width = 1
    # x_c = 'g'
    # y_c = 'y'
    # z_c = 'k'
    # ax.plot([-axis_size, axis_size], [0, 0], [0, 0], c=z_c, linewidth=line_width)
    # ax.plot([0, 0], [-axis_size, axis_size], [0, 0], c=z_c, linewidth=line_width)
    # ax.plot([0, 0], [0, 0], [-axis_size, axis_size], c=z_c, linewidth=line_width)

def translate_face_to_origin(anchor_point, face_matrix):
    x = face_matrix['x'].values
    y = face_matrix['y'].values
    z = face_matrix['z'].values

    anchor_x = x[anchor_point - 1]
    anchor_y = y[anchor_point - 1]
    anchor_z = z[anchor_point - 1]

    x = x - anchor_x
    y = y - anchor_y
    z = z - anchor_z

    face_matrix_copy = face_matrix.copy()
    face_matrix_copy['x'] = x
    face_matrix_copy['y'] = y
    face_matrix_copy['z'] = z
    return face_matrix_copy

def rotate_face_to_match_z_axis(sub_anchor_point, face_matrix):
    global _180_above_pi
    sap = face_matrix[face_matrix['point'] == sub_anchor_point].values[0]
    sap_copy = sap[1:]

    angle_to_rotate_x_axis = 90.0 - math.atan2(sap_copy[2], sap_copy[1]) * _180_above_pi
    new_sap = rotate_around_axis(angle_to_rotate_x_axis, sap_copy, axis='x')
    
    angle_to_rotate_y_axis = math.atan2(new_sap[2], new_sap[0]) * _180_above_pi - 90.0
    new_sap = rotate_around_axis(angle_to_rotate_y_axis, new_sap, axis='y')

    face_matrix.loc[face_matrix['point'] == sub_anchor_point, 'x'] = new_sap[0]
    face_matrix.loc[face_matrix['point'] == sub_anchor_point, 'y'] = new_sap[1]
    face_matrix.loc[face_matrix['point'] == sub_anchor_point, 'z'] = new_sap[2]

    point_list = face_matrix['point']
    x_list = face_matrix['x']
    y_list = face_matrix['y']
    z_list = face_matrix['z']

    new_face_matrix = []
    for point in zip(point_list, x_list, y_list, z_list):
        point_xyz = np.array(point[1:])

        if point[0] != 28 and point[0] != 29:
            point_xyz = rotate_around_axis(angle_to_rotate_x_axis, point_xyz, axis='x')
            point_xyz = rotate_around_axis(angle_to_rotate_y_axis, point_xyz, axis='y')

        new_face_matrix.append([point[0], point_xyz[0], point_xyz[1], point_xyz[2]])

    new_face_matrix = pd.DataFrame(new_face_matrix, columns=['point', 'x', 'y', 'z'])

    return new_face_matrix

def find_point_projection_on_line(point: tuple, line_point1: tuple, line_point2: tuple):
    point = geometry.Point(point[0], point[0])
    line = geometry.LineString([line_point1, line_point2])

    x = np.array(point.coords[0])
    u = np.array(line.coords[0])
    v = np.array(line.coords[1])

    n = v - u
    n /= np.linalg.norm(n, 2)
    P = u + n*np.dot(x - u, n)

    return P

def rotate_face_to_parallel_x_axis(left_anchor_point, right_anchor_point, face_matrix):
    global _180_above_pi
    lap = face_matrix[face_matrix['point'] == left_anchor_point].values[0]
    rap = face_matrix[face_matrix['point'] == right_anchor_point].values[0]

    lap = lap[1:]
    rap = rap[1:]

    point_projection = find_point_projection_on_line(point=(0, 0), line_point1=(lap[0], lap[1]), line_point2=(rap[0], rap[1]))
    angle_to_rotate_z_axis = 90.0 - math.atan2(point_projection[1], point_projection[0]) * _180_above_pi

    point_list = face_matrix['point']
    x_list = face_matrix['x']
    y_list = face_matrix['y']
    z_list = face_matrix['z']

    new_face_matrix = []
    for point in zip(point_list, x_list, y_list, z_list):
        point_xyz = np.array(point[1:])

        if point[0] != 28 and point[0] != 29:
            point_xyz = rotate_around_axis(angle_to_rotate_z_axis, point_xyz, axis='z')

        new_face_matrix.append([point[0], point_xyz[0], point_xyz[1], point_xyz[2]])

    new_face_matrix = pd.DataFrame(new_face_matrix, columns=['point', 'x', 'y', 'z'])

    return new_face_matrix
    