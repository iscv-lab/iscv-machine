from sklearn.preprocessing import MinMaxScaler

import os, sys
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial


sys.path.append(os.path.abspath(os.path.join("..", "..")))

O_scaler = MinMaxScaler(feature_range=(-3.2, 2.3))
C_scaler = MinMaxScaler(feature_range=(-2.5, 2.4))
E_scaler = MinMaxScaler(feature_range=(-3.6, 2.5))
A_scaler = MinMaxScaler(feature_range=(-3.0, 2.1))
N_scaler = MinMaxScaler(feature_range=(-3.0, 2.0))


def ScoreMinMaxScaler(result, type):
    score = [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16],
        [17],
        [18],
        [19],
        [20],
        [21],
        [22],
        [23],
        [24],
        [25],
        [26],
        [27],
        [28],
        [29],
        [30],
        [31],
        [32],
        [33],
        [34],
        [35],
        [36],
        [37],
        [38],
        [39],
        [40],
    ]

    # Openness Scaler
    if type == "Openness":
        # O_scaler=MinMaxScaler(feature_range=(-3.2, 2.3))
        O_score_data = O_scaler.fit_transform(score)
        O_score_scale = O_score_data[result]
        return O_score_scale

    # Conscientiousness Scaler
    if type == "Conscientiousness":
        # C_scaler=MinMaxScaler(feature_range=(-2.5, 2.4))
        C_score_data = C_scaler.fit_transform(score)
        C_score_scale = C_score_data[result]
        return C_score_scale

    # Extroversion Scaler
    if type == "Extroversion":
        # E_scaler=MinMaxScaler(feature_range=(-3.6, 2.5))
        E_score_data = E_scaler.fit_transform(score)
        E_score_scale = E_score_data[result]
        return E_score_scale

    # Agreeableness Scaler
    if type == "Agreeableness":
        # A_scaler=MinMaxScaler(feature_range=(-3.0, 2.1))
        A_score_data = A_scaler.fit_transform(score)
        A_score_scale = A_score_data[result]
        return A_score_scale

    # Neuroticism Scaler
    if type == "Neuroticism":
        # N_scaler=MinMaxScaler(feature_range=(-3.0, 2.0))
        N_score_data = N_scaler.fit_transform(score)
        N_score_scale = N_score_data[result]
        return N_score_scale


def BigFiveFormula(calculation):
    # Extroversion
    E_score = (
        20
        + calculation[0]
        - calculation[5]
        + calculation[10]
        - calculation[15]
        + calculation[20]
        - calculation[25]
        + calculation[30]
        - calculation[35]
        + calculation[40]
        - calculation[45]
    )

    # E_score_scale = ScoreMinMaxScaler(E_score, "Extroversion")

    # Agreeableness
    A_score = (
        14
        - calculation[1]
        + calculation[6]
        - calculation[11]
        + calculation[16]
        - calculation[21]
        + calculation[26]
        - calculation[31]
        + calculation[36]
        + calculation[41]
        + calculation[46]
    )

    # A_score_scale = ScoreMinMaxScaler(A_score, "Agreeableness")

    # Conscientiousness
    C_score = (
        14
        + calculation[2]
        - calculation[7]
        + calculation[12]
        - calculation[17]
        + calculation[22]
        - calculation[27]
        + calculation[32]
        - calculation[37]
        + calculation[42]
        + calculation[47]
    )

    # C_score_scale = ScoreMinMaxScaler(C_score, "Conscientiousness")

    # Neuroticism
    N_score = (
        38
        - calculation[3]
        + calculation[8]
        - calculation[13]
        + calculation[18]
        - calculation[23]
        - calculation[28]
        - calculation[33]
        - calculation[38]
        - calculation[43]
        - calculation[48]
    )

    # N_score_scale = ScoreMinMaxScaler(N_score, "Neuroticism")

    # Openness
    O_score = (
        8
        + calculation[4]
        - calculation[9]
        + calculation[14]
        - calculation[19]
        + calculation[24]
        - calculation[29]
        + calculation[34]
        + calculation[39]
        + calculation[44]
        + calculation[49]
    )

    # O_score_scale = ScoreMinMaxScaler(O_score, "Openness")

    OCEAN_score_scale = []
    OCEAN_score_scale.append(E_score)
    OCEAN_score_scale.append(A_score)
    OCEAN_score_scale.append(C_score)
    OCEAN_score_scale.append(N_score)
    OCEAN_score_scale.append(O_score)
    return OCEAN_score_scale


################################################################################################


def to_result_txt(Result: list, Comment: list, file_result_path: str):
    with open(file_result_path, "w", encoding="utf-8") as file:
        file.write("Extroversion Score: " + str(int(Result[0][0])) + "/40\n")
        file.write(str(Comment[0]) + "\n")
        file.write("Agreeableness Score: " + str(int(Result[1][0])) + "/40\n")
        file.write(str(Comment[1]) + "\n")
        file.write("Conscientiousness Score: " + str(int(Result[2][0])) + "/40\n")
        file.write(str(Comment[2]) + "\n")
        file.write("Neuroticism Score: " + str(int(Result[3][0])) + "/40\n")
        file.write(str(Comment[3]) + "\n")
        file.write("Openness to Experience Score: " + str(int(Result[4][0])) + "/40\n")
        file.write(str(Comment[4]) + "\n")


# ########


def to_result_json(Result: list, file_result_path: str):
    # Prepare the data to be saved as JSON
    data = {
        "e": int(Result[0]),
        # "ec": (Comment[0].replace("Extroversion Comment: ", "")),
        "a": int(Result[1]),
        # "ac": (Comment[1].replace("Agreeableness Comment: ", "")),
        "c": int(Result[2]),
        # "cc": (Comment[2].replace("Conscientiousness Comment: ", "")),
        "n": int(Result[3]),
        # "nc": (Comment[3].replace("Neuroticism Comment: ", "")),
        "o": int(Result[4]),
        # "oc": (Comment[4].replace("Openness to Experience Comment: ", "")),
    }

    # Save the data as JSON
    with open(file_result_path, "w", encoding="utf-8") as file:
        json.dump(data, file, allow_nan=False)
    return data


# =============================================================================


def handle_big_five_audio(session_id: str):
    file_qa_path = f"./public/interview/{session_id}/"
    with open(file_qa_path + "qa.txt", "r") as file:
        # Initialize an empty array
        values = []

        # Read the file line by line and append each numeric value to the array
        for line in file:
            numeric_value = int(line.strip())  # Convert line to a float
            values.append(numeric_value)

    # drawGraph(Avg_Inverse_Result(BigFiveFormula(values)), 1, file_qa_path)

    file_result_json_path = file_qa_path + "audio.json"
    Result = BigFiveFormula(values)
    return to_result_json(Result, file_result_json_path)
