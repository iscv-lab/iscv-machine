from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
from tensorflow import keras

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def load_recommend_model():
    filename = "./IT_Job_Data_Clean.csv"
    data = pd.read_csv(os.path.join(CURRENT_PATH, filename), encoding="unicode_escape")
    model = keras.models.load_model(os.path.join(CURRENT_PATH, "model.h5"))
    print("loaded recommend model")
    return data, model


def remove_duplicates(arr):
    result = []
    for num in arr:
        if num not in result:
            result.append(num)
    return result


def recommended_jobs(arr_skill, data, model):
    list_skills = ",".join(arr_skill)
    vocab_size = 100
    sequences_length = 120

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data["Require"])

    user_profile_sequence = tokenizer.texts_to_sequences([list_skills])
    user_profile_padded = pad_sequences(
        user_profile_sequence, maxlen=sequences_length, padding="post"
    )

    predictions = model.predict(user_profile_padded)
    job_data = data["JobTitle"]
    predict_jobs = np.argsort(predictions[0])[::-1]
    predict_jobs_title = [job_data[i] for i in predict_jobs]
    unique_jobs = remove_duplicates(predict_jobs_title)
    jobs_result = unique_jobs[:10]

    # Display the top-k recommended job listings

    return jobs_result
