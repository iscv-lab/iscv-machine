import sys
import os
import argparse
import imageio
from tqdm import tqdm
import yaml
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..Face.FaceBoxes import FaceBoxes
from ..Face.TDDFA import TDDFA
from ..Face.utils.render import render
from ..Face.utils.functions import cv_draw_landmark, get_suffix, draw_landmarks
from ..Face.utils.depth import depth
from ..Face.utils.pncc import pncc
from ..Face.utils.uv import uv_tex
from ..Face.utils.pose import viz_pose
from ..Face.utils.serialization import ser_to_ply, ser_to_obj
from ..Face.utils.tddfa_util import str2bool

from .video_quality import variance_of_laplacian, mean_percent_bright

from skimage.exposure import is_low_contrast


def detect_3d_landmarks_video(
    video_fp,
    config: str = "configs/mb1_120x120.yml",
    mode: str = "cpu",
    opt: str = "2d_sparse",
    onnx: bool = True,
    export_video_result: bool = False,
):
    opt_type = ["2d_sparse", "3d"]
    if opt not in opt_type:
        raise ValueError("Invalid opt. Expected one of: %s" % opt_type)

    mode_type = ["cpu", "gpu"]
    if mode not in mode_type:
        raise ValueError("Invalid mode. Expected one of: %s" % mode_type)

    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if onnx:
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = mode == "gpu"
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    fn = video_fp.split("/")[-1]
    reader = imageio.get_reader(video_fp)

    fps = reader.get_meta_data()["fps"]

    suffix = get_suffix(video_fp)
    video_wfp = f'examples/results/result_{fn.replace(suffix, "")}_{opt}.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)

    # Init list to store all verbose data
    landmarks_data = []

    # run
    dense_flag = opt in ("3d",)
    pre_ver = None

    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            # print(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy="landmark")
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy="landmark")

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        pre_ver = ver  # for tracking

        landmarks_data.append(ver)

        if export_video_result == True:
            if opt == "2d_sparse":
                res = cv_draw_landmark(frame_bgr, ver)
            elif opt == "3d":
                res = render(frame_bgr, [ver], tddfa.tri)
            writer.append_data(res[..., ::-1])  # BGR->RGB

    writer.close()
    if export_video_result == True:
        print(f"Path to video result: {video_wfp}")

    return landmarks_data


def detect_3d_landmarks_image(
    img_fp,
    config: str = "configs/mb1_120x120.yml",
    mode: str = "cpu",
    opt: str = "2d_sparse",
    onnx: bool = True,
    export_image_result: bool = False,
    show_flag="true",
):
    mode_type = ["cpu", "gpu"]
    if mode not in mode_type:
        raise ValueError("Invalid mode. Expected one of: %s" % mode_type)

    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if onnx:
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = mode == "gpu"
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f"No face detected, exit")
        sys.exit(-1)
    print(f"Detect {n} faces")

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = opt in ("2d_dense", "3d", "depth", "pncc", "uv_tex", "ply", "obj")
    old_suffix = get_suffix(img_fp)
    new_suffix = f".{opt}" if opt in ("ply", "obj") else ".jpg"

    wfp = (
        f'examples/results/result_{img_fp.split("/")[-1].replace(old_suffix, "")}_{opt}'
        + new_suffix
    )

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    if export_image_result == True:
        if opt == "2d_sparse":
            draw_landmarks(
                img, ver_lst, show_flag=show_flag, dense_flag=dense_flag, wfp=wfp
            )
        elif opt == "2d_dense":
            draw_landmarks(
                img, ver_lst, show_flag=show_flag, dense_flag=dense_flag, wfp=wfp
            )
        elif opt == "3d":
            render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=show_flag, wfp=wfp)
        elif opt == "depth":
            # if `with_bf_flag` is False, the background is black
            depth(
                img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp, with_bg_flag=True
            )
        elif opt == "pncc":
            pncc(
                img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp, with_bg_flag=True
            )
        elif opt == "uv_tex":
            uv_tex(img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp)
        elif opt == "pose":
            viz_pose(img, param_lst, ver_lst, show_flag=show_flag, wfp=wfp)
        elif opt == "ply":
            ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
        elif opt == "obj":
            ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
        else:
            raise ValueError(f"Unknown opt {opt}")

    return ver_lst


def detect_3d_with_quality_assessment(
    video_fp,
    max_frame=00,
    max_error=100,
    min_bright_percent=0.37,
    contrast_thresh=0.3,
    blurry_thresh=100,
    skip_from_frame=-1,
    config: str = "configs/mb1_120x120.yml",
    mode: str = "cpu",
    opt: str = "2d_sparse",
    onnx: bool = True,
    export_video_result: bool = False,
):
    opt_type = ["2d_sparse", "3d"]
    if opt not in opt_type:
        raise ValueError("Invalid opt. Expected one of: %s" % opt_type)

    mode_type = ["cpu", "gpu"]
    if mode not in mode_type:
        raise ValueError("Invalid mode. Expected one of: %s" % mode_type)

    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if onnx:
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = mode == "gpu"
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    fn = video_fp.split("/")[-1]
    reader = imageio.get_reader(video_fp)

    fps = reader.get_meta_data()["fps"]

    suffix = get_suffix(video_fp)
    video_wfp = f'examples/results/result_{fn.replace(suffix, "")}_{opt}.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)

    # Init list to store all verbose data
    landmarks_data = []

    # run
    dense_flag = opt in ("3d",)
    pre_ver = None

    num_error = 0
    num_contrast = 0
    num_blurry = 0
    num_non_face = 0
    num_dark_face = 0
    frame_count = 0

    for i, frame in tqdm(enumerate(reader)):
        if i <= skip_from_frame:
            continue
        if num_error >= max_error:
            break

        frame_bgr = frame[..., ::-1]  # RGB->BGR
        gray_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Checkcontrast
        if is_low_contrast(gray_image, fraction_threshold=contrast_thresh):
            num_error += 1
            num_contrast += 1
            if export_video_result == True:
                writer.append_data(frame_bgr[..., ::-1])  # BGR->RGB
            continue

        # Check blurry
        fm = variance_of_laplacian(gray_image)
        if fm < blurry_thresh:
            num_error += 1
            num_blurry += 1
            if export_video_result == True:
                writer.append_data(frame_bgr[..., ::-1])  # BGR->RGB
            continue

        # Face detection with Faceboxes
        boxes = face_boxes(frame_bgr)

        # Check non face
        if len(boxes) <= 0:
            num_error += 1
            num_non_face += 1
            if export_video_result == True:
                writer.append_data(frame_bgr[..., ::-1])  # BGR->RGB
            continue

        # Get facial image
        (sx, sy, ex, ey) = np.round(boxes[0][:4]).astype(np.int32)
        face_img = gray_image[sy:ey, sx:ex]

        # Check brightness
        bright_percent = mean_percent_bright(image=face_img)
        if bright_percent <= min_bright_percent:
            num_error += 1
            num_dark_face += 1
            if export_video_result == True:
                writer.append_data(frame_bgr[..., ::-1])  # BGR->RGB
            continue

        # print(frame_bgr)
        boxes = [boxes[0]]
        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # refine
        param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy="landmark")
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        landmarks_data.append(ver)

        if export_video_result == True:
            if opt == "2d_sparse":
                res = cv_draw_landmark(frame_bgr, ver, size=2)
            elif opt == "3d":
                res = render(frame_bgr, [ver], tddfa.tri)
            writer.append_data(res[..., ::-1])  # BGR->RGB

        frame_count += 1
        if frame_count >= max_frame:
            break

    writer.close()

    if num_error >= max_error:
        print("ERROR: Video has more than {0} errors".format(max_error))
        print("\t- {0} contrast".format(num_contrast))
        print("\t- {0} blurry".format(num_blurry))
        print("\t- {0} non-face".format(num_non_face))
        print("\t- {0} dark face".format(num_dark_face))
        return None

    if export_video_result == True:
        print(f"Path to video result: {video_wfp}")

    return landmarks_data
