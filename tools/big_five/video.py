import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import argparse
from BigFiveVisualModel import BigFiveVisualModel

# Create an argument parser
parser = argparse.ArgumentParser(description='Demo Big Five Script')

# Add the input parameters
parser.add_argument('--param1', type=int, help='Parameter 1 description')
parser.add_argument('--param2', type=str, help='Parameter 2 description')

# Parse the input arguments
args = parser.parse_args()

# Access the input parameters
param1_value = args.param1

# Use the input parameters in your script
print("Parameter 1:", param1_value)

MIN_BRIGHT_PERCENT = 0.37
CONTRAST_THRESH = 0.3
BLURRY_THRESH = 100

model = BigFiveVisualModel(model_path="saved_model/CNN_LSTM_model.h5", scale_min=0, scale_max=40)


vid = 'examples/inputs/67_video.mp4'


bigfive_result = model.predict_bigfive(video_path=vid,
                                    skip_from_frame = -1, # -1 là ko skip gì hết 
                                    max_frame = 500, # lấy 500 frame để predict 
                                    max_error = 100, # cho phép tối đa 100 frame bị lỗi 
                                    min_bright_percent = MIN_BRIGHT_PERCENT, 
                                    contrast_thresh = CONTRAST_THRESH, 
                                    blurry_thresh = BLURRY_THRESH)
print('------------------------------\nResult of video {0}:'.format(vid))
print('O-C-E-A-N:', bigfive_result)
print('------------------------------\n')