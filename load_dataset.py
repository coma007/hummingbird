# package for music and audio analysis
import librosa
import numpy as np
import os
# for progress visualization
from tqdm import tqdm
# for csv files
import pandas as pd


def mfccs_feature_extractor(file):

    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Scaled mfccs to its mean
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


def load_dataset(path):
    # print(path)
    # abs_path = os.path.abspath(path)
    # print(abs_path)

    # labels and feature
    data = []

    new_path = os.path.join(path, "MLEndHWD_Audio_Attributes.csv")
    load_metadata(new_path, data)
    for folder in tqdm(os.listdir(path)):
        if ".csv" in folder:
            continue
        for file in os.listdir(os.path.join(path, folder)):
            features = mfccs_feature_extractor(
                os.path.join(path, folder, file))
            index = int(file[:-4])
            data[index].append(features)
    return data


def load_metadata(path, data):
    df = pd.read_csv(path)
    for i in range(len(df[list(df.keys())[0]])):
        data.append([])
        for column in df:
            data[i].append(df[column][i])
    # print(data[0])
