import sys, os
sys.path.append(os.path.abspath(os.path.join('3DDFA_V2')))

from myutils.detect3d import detect_3d_landmarks_video, detect_3d_landmarks_image, detect_3d_with_quality_assessment

# res = detect_3d_with_quality_assessment(video_fp="examples/inputs/67_video.mp4", export_video_result=True, 
#                                         max_frame=200, 
#                                         max_error=100, 
#                                         min_bright_percent=0.37, 
#                                         contrast_thresh=0.3, 
#                                         blurry_thresh=100,
#                                         skip_from_frame=90)
res = detect_3d_with_quality_assessment(video_fp="examples/inputs/1_video.mp4", export_video_result=True, 
                                        max_frame=500, 
                                        max_error=100, 
                                        min_bright_percent=0.37, 
                                        contrast_thresh=0.3, 
                                        blurry_thresh=100,
                                        skip_from_frame=100)
if res != None:
    print(len(res))
    print(len(res[0]))
    print(len(res[0][0]))