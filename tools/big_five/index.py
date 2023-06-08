import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
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

    E_score_scale = ScoreMinMaxScaler(E_score, "Extroversion")

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

    A_score_scale = ScoreMinMaxScaler(A_score, "Agreeableness")

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

    C_score_scale = ScoreMinMaxScaler(C_score, "Conscientiousness")

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

    N_score_scale = ScoreMinMaxScaler(N_score, "Neuroticism")

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

    O_score_scale = ScoreMinMaxScaler(O_score, "Openness")

    OCEAN_score_scale = []
    OCEAN_score_scale.append(E_score_scale[0])
    OCEAN_score_scale.append(A_score_scale[0])
    OCEAN_score_scale.append(C_score_scale[0])
    OCEAN_score_scale.append(N_score_scale[0])
    OCEAN_score_scale.append(O_score_scale[0])

    return OCEAN_score_scale


def Avg_Inverse_Result(score_scale):
    PEcv = -0.1
    PAcv = -0.32
    PCcv = -0.173
    PNcv = -0.88
    POcv = -0.174

    # Average Result
    # E_result=(score_scale[0]+PEcv)/2
    # A_result=(score_scale[1]+PAcv)/2
    # C_result=(score_scale[2]+PCcv)/2
    # N_result=(score_scale[3]+PNcv)/2
    # O_result=(score_scale[4]+POcv)/2

    E_result = score_scale[0]
    A_result = score_scale[1]
    C_result = score_scale[2]
    N_result = score_scale[3]
    O_result = score_scale[4]

    # Inverse scale
    E_final = E_scaler.inverse_transform([[E_result], [0]])
    A_final = A_scaler.inverse_transform([[A_result], [0]])
    C_final = C_scaler.inverse_transform([[C_result], [0]])
    N_final = N_scaler.inverse_transform([[N_result], [0]])
    O_final = O_scaler.inverse_transform([[O_result], [0]])

    OCEAN_final = []
    OCEAN_final.append(E_final[0])
    OCEAN_final.append(A_final[0])
    OCEAN_final.append(C_final[0])
    OCEAN_final.append(N_final[0])
    OCEAN_final.append(O_final[0])

    return OCEAN_final


def radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """Draw. If frame is polygon, make gridlines polygon-shaped"""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def drawGraph(OCEAN_score, index, path):
    data = [
        [
            "Extroversion",
            "Agreeableness",
            "Conscientiousness",
            "Neuroticism",
            "Openness",
        ],
        (
            "The Big Five Personality Score",
            [
                [
                    OCEAN_score[0],
                    OCEAN_score[1],
                    OCEAN_score[2],
                    OCEAN_score[3],
                    OCEAN_score[4],
                ]
            ],
        ),
    ]

    N = len(data[0])
    theta = radar_factory(N, frame="polygon")  # polygon  !!!

    spoke_labels = data.pop(0)
    title, case_data = data[0]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids([0, 10, 20, 30, 40])
    ax.set_title(title, position=(0.5, 1.1), ha="center")
    ax.set_ylim(0, 40)
    for d in case_data:
        line = ax.plot(theta, d)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    plt.savefig(path + "chart.png")
    plt.close()
    # plt.show()


def BigFiveComment(result, current_path):
    # Lưu số điểm từng tính cách vào dataframe
    df_result = pd.DataFrame(
        (
            [
                ["Extroversion", result[0]],
                ["Agreeableness", result[1]],
                ["Conscientiousness", result[2]],
                ["Neuroticism", result[3]],
                ["Openness to Experience", result[4]],
            ]
        ),
        columns=["Type", "Result"],
    )
    # Lưu dữ liệu đánh giá tính cách vào dataframe
    df_comment = pd.read_csv(current_path + "comments.csv")

    # Merge 2 dataframe
    df_merge = pd.merge(df_comment, df_result, on="Type", how="outer")
    # Lọc ra những dòng dữ liệu thỏa điều kiện
    df_final = df_merge[
        (df_merge["Result"] >= df_merge["StartScore"])
        & (df_merge["Result"] <= df_merge["EndScore"])
    ].reset_index()
    # Hiển thị đánh giá
    comments = []
    for i in range(5):
        comments.append(df_final["Type"][i] + " Comment: " + df_final["Comment"][i])
    return comments


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


def to_json_txt(Result: list, Comment: list, file_result_path: str):
    # Prepare the data to be saved as JSON
    data = {
        "Extroversion Score": int(Result[0][0]),
        "Extroversion Comment": Comment[0].replace("Extroversion Comment: ", ""),
        "Agreeableness Score": int(Result[1][0]),
        "Agreeableness Comment": Comment[1].replace("Agreeableness Comment: ", ""),
        "Conscientiousness Score": int(Result[2][0]),
        "Conscientiousness Comment": Comment[2].replace(
            "Conscientiousness Comment: ", ""
        ),
        "Neuroticism Score": int(Result[3][0]),
        "Neuroticism Comment": Comment[3].replace("Neuroticism Comment: ", ""),
        "Openness to Experience Score": int(Result[4][0]),
        "Openness to Experience Comment": Comment[4].replace(
            "Openness to Experience Comment: ", ""
        ),
    }

    # Save the data as JSON
    with open(file_result_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
    return data


# =============================================================================


def handle_big_five(interview_id: str):
    executor = ThreadPoolExecutor()
    current_path = os.path.dirname(__file__) + "/"
    file_qa_path = f"./public/interview/{interview_id}/"
    with open(file_qa_path + "qa.txt", "r") as file:
        # Initialize an empty array
        values = []

        # Read the file line by line and append each numeric value to the array
        for line in file:
            numeric_value = int(line.strip())  # Convert line to a float
            values.append(numeric_value)

    drawGraph(Avg_Inverse_Result(BigFiveFormula(values)), 1, file_qa_path)

    file_result_txt_path = file_qa_path + "result.txt"
    file_result_json_path = file_qa_path + "result.json"

    Result = Avg_Inverse_Result(BigFiveFormula(values))
    Comment = BigFiveComment(Avg_Inverse_Result(BigFiveFormula(values)), current_path)
    txt_task = executor.submit(
        partial(to_result_txt, Result, Comment, file_result_txt_path)
    )
    json_task = executor.submit(
        partial(to_json_txt, Result, Comment, file_result_json_path)
    )
    txt_task.result()
    return json_task.result()
