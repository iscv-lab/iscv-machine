from call_funtions import load_model, predict_bigfive

# Call load_model function to load model 
model = load_model()

videos = [
    'examples/inputs/67_video.mp4'
]

for vid in videos:
    # Call predict_bigfive function to perform prediction 
    bigfive_result = predict_bigfive(model=model, 
                                    video_path=vid, 
                                    frame_num_to_extract = 700, 
                                    start_from_frame = 100,
                                    mode = 'cpu')

    print('------------------------------\nResult of video {0}:'.format(vid))
    print(bigfive_result)
    print('------------------------------\n')