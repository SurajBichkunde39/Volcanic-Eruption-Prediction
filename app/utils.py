import requests
import os
import random
import string

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_files_in():
    # https://drive.google.com/file/d/1-20DtjI_4E_RkY-WlFAbnNa7LTtQhs1C/view?usp=sharing
    # https://drive.google.com/file/d/1zF4UOMHTz8k4tjL7ZnzSkK2qnxfmrteM/view?usp=sharing
    # final_columns file
    # new file id
    # https://drive.google.com/file/d/1Y_eyFpucRaCis1mbPcMuuirKi03P1Qa4/view?usp=sharing
    file_id = '1-20DtjI_4E_RkY-WlFAbnNa7LTtQhs1C'
    destination = "./app/final_columns.pkl"
    download_file_from_google_drive(file_id, destination)
    # best model file
    # https://drive.google.com/file/d/1zF4UOMHTz8k4tjL7ZnzSkK2qnxfmrteM/view?usp=sharing
    file_id = '1zF4UOMHTz8k4tjL7ZnzSkK2qnxfmrteM'
    destination = "./app/best_model.json"
    download_file_from_google_drive(file_id, destination)
    # # trying pickel file
    # https://drive.google.com/file/d/1-9z1aYFGlgbKGewnn5MMcA8QuQV6WNMV/view?usp=sharing
    # file_id = '1-9z1aYFGlgbKGewnn5MMcA8QuQV6WNMV'
    # destination = "./app/best_model.pkl"
    # download_file_from_google_drive(file_id, destination)
    # done


def get_files():
    all_files = os.listdir()
    if 'best_model.josn' not in all_files:
        get_files_in()


def generate_ranom_string(N=5):
    res = ''.join(
        random.choices(
            string.ascii_uppercase +
            string.digits, k=N
        )
    )
    return res


def check_dataframe(df):
    # check for the shape
    check1 = df.shape == (60001, 10)
    # check for the names of the columns
    check2 = True
    expected_sensors = []
    for i in range(1, 11):
        expected_sensors.append(f'sensor_{i}')
    check2_cols = list(df.columns)
    for a, b in zip(expected_sensors, check2_cols):
        if a != b:
            check2 = False
    # prepare the error message
    msg = {"all_ok": "okay"}
    if not check1:
        error_str = f"Expected shape = (60001,10), got {df.shape}"
        msg["shape_miss_match"] = error_str
        msg["all_ok"] = "no"
    if not check2:
        e_str = f"Expected columns {expected_sensors},got {list(df.columns)}"
        msg["column_name_miss"] = e_str
        msg["all_ok"] = "no"
    return msg


def preprocess_dataframe(df):
    is_sensor_missing = None
    percentage_sensor_missing = []
    temp_df = df
    sensor_wise_null = temp_df.isnull().sum(axis=0)
    if sensor_wise_null.sum() == 0:
        is_sensor_missing = 0
    else:
        is_sensor_missing = 1
    index = 1
    for miss in sensor_wise_null:
        percentage_miss = miss / len(temp_df)
        percentage_sensor_missing.append(percentage_miss)
        index += 1

    missing_info = [is_sensor_missing]
    missing_info.extend(percentage_sensor_missing)

    # handle the missing values
    temp_df.fillna(value=0, inplace=True)

    # calculate the basic features
    sensors = []
    for i in range(1, 11):
        sensors.append(f'sensor_{i}_')

    check = temp_df
    basic_features = []
    basic_features.extend(list(check.sum().ravel()))
    basic_features.extend(list(check.min().ravel()))
    basic_features.extend(list(check.max().ravel()))
    basic_features.extend(list(check.mean().ravel()))
    basic_features.extend(list(check.median().ravel()))
    basic_features.extend(list(check.std().ravel()))
    basic_features.extend(list(check.skew().ravel()))
    basic_features.extend(list(check.kurtosis().ravel()))

    q_features = []
    # add quantiles for actual values
    q_features.extend(list(check.quantile(.99).ravel()))
    q_features.extend(list(check.quantile(.95).ravel()))
    q_features.extend(list(check.quantile(.90).ravel()))
    q_features.extend(list(check.quantile(.85).ravel()))
    q_features.extend(list(check.quantile(.75).ravel()))
    q_features.extend(list(check.quantile(.50).ravel()))
    q_features.extend(list(check.quantile(.25).ravel()))
    q_features.extend(list(check.quantile(.15).ravel()))
    q_features.extend(list(check.quantile(.10).ravel()))
    q_features.extend(list(check.quantile(.05).ravel()))

    # add quantile for abs values
    check = np.abs(check)
    q_features.extend(list(check.quantile(.99).ravel()))
    q_features.extend(list(check.quantile(.95).ravel()))
    q_features.extend(list(check.quantile(.90).ravel()))
    q_features.extend(list(check.quantile(.85).ravel()))
    q_features.extend(list(check.quantile(.75).ravel()))
    q_features.extend(list(check.quantile(.50).ravel()))
    q_features.extend(list(check.quantile(.25).ravel()))
    q_features.extend(list(check.quantile(.15).ravel()))
    q_features.extend(list(check.quantile(.10).ravel()))
    q_features.extend(list(check.quantile(.05).ravel()))

    # FFT fartures
    check = temp_df
    fft_res = np.fft.fft(check)
    fft_real = np.real(fft_res)
    fft_imag = np.imag(fft_res)

    fft_features = list(abs(fft_res[0]))
    # ..
    # stats of the real valued data
    fft_features.extend(list(fft_real.min(axis=0)))
    fft_features.extend(list(fft_real.max(axis=0)))
    fft_features.extend(list(fft_real.mean(axis=0)))
    fft_features.extend(list(np.median(fft_real, axis=0)))
    fft_features.extend(list(stats.skew(fft_real)))
    fft_features.extend(list(stats.kurtosis(fft_real)))

    # stats from the imagnary component
    fft_features.extend(list(fft_imag.min(axis=0)))
    fft_features.extend(list(fft_imag.max(axis=0)))
    fft_features.extend(list(fft_imag.mean(axis=0)))
    fft_features.extend(list(np.median(fft_imag, axis=0)))
    fft_features.extend(list(stats.skew(fft_imag)))
    fft_features.extend(list(stats.kurtosis(fft_imag)))

    final_all = []
    final_all.extend(basic_features)
    final_all.extend(q_features)
    final_all.extend(fft_features)
    final_all.extend(missing_info)
    # return all the calculated features
    return final_all


def plot_the_sensors(temp_df, filename):
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 10))
    axes = axes.ravel()
    for cur_ax, signal, c in zip(axes, temp_df.columns, colors):
        cur_ax.plot(temp_df[signal], label=signal, c=c)
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    fig.legend(loc="upper right", title="segment")
    fig.tight_layout()
    fig.subplots_adjust(right=0.89)
    plt.savefig(filename)
