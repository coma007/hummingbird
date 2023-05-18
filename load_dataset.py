import librosa
import numpy as np
import os
from tqdm import tqdm
import pandas as pd


def load_dataset(path):
    data = []

    new_path = os.path.join(path, "MLEndHWD_Audio_Attributes.csv")
    load_metadata(new_path, data)
    # for folder in tqdm(os.listdir(path)):
    #     if ".csv" in folder:
    #         continue
    #     for file in os.listdir(os.path.join(path, folder)):
    #         features = mfccs_feature_extractor(
    #             os.path.join(path, folder, file))
    #         index = int(file[:-4])
    #         data[index].append(features)
    data_frame = pd.DataFrame(
        data, columns=['song_file', 'interpreter', 'label', 'audio_type'])
    return data_frame


def load_metadata(path, data):
    df = pd.read_csv(path)
    for i in range(len(df[list(df.keys())[0]])):
        data.append([])
        for column in df:
            data[i].append(df[column][i])


def mfccs_feature_extractor(file):

    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Scaled mfccs to its mean
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features
