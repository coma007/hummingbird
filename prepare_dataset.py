from sklearn.preprocessing import StandardScaler
from numpy import where
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import tqdm
import librosa


def prepare_dataset(data, filepaths):
    X, y, x = get_features(filepaths, labels_file=data,
                           scale_audio=True, onlySingleDigit=True)
    scale_data(X, y, x)
    counter = Counter(y)
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()


def get_pitch(x, fs, winLen=0.02):
    # winLen = 0.02
    p = winLen*fs
    frame_length = int(2**int(p-1).bit_length())
    hop_length = frame_length//2
    f0, voiced_flag, voiced_probs = librosa.pyin(y=x, fmin=80, fmax=450, sr=fs,
                                                 frame_length=frame_length, hop_length=hop_length)
    return f0, voiced_flag


def get_features(files, labels_file, scale_audio=False, onlySingleDigit=False):
    # different song lable
    slable = ['Frozen', 'Harry', 'Panther',
              'StarWars', 'Rain', 'Hakuna', 'Mamma']
    X, y, z = [], [], []
    for file in tqdm.tqdm(files):
        features = []
        fileID = int(file.split('/')[-1].split('.')[0])
        file_name = file.split('/')[-1]

        # Feature for interpretation type of the file, True for hum, False for whistle
        y1 = labels_file.loc[fileID]['audio_type'] == 'hum'

        # Feature for type of the song,
        if (labels_file.loc[fileID]['label'] == slable[0]):
            y2 = 0  # label 0 if file is Frozen song
        elif (labels_file.loc[fileID]['label'] == slable[1]):
            y2 = 1  # label 1 if file is Harry song
        elif (labels_file.loc[fileID]['label'] == slable[2]):
            y2 = 2  # label 2 if file is Panther song
        elif (labels_file.loc[fileID]['label'] == slable[3]):
            y2 = 3  # label 3 if file is Starwars song
        elif (labels_file.loc[fileID]['label'] == slable[4]):
            y2 = 4  # label 4 if file is Raina song
        elif (labels_file.loc[fileID]['label'] == slable[5]):
            y2 = 5  # label 5 if file is Hakuna song
        else:
            y2 = 6  # label 6 if file is Mamma song

        # print('Y2 is :',y2)

        fs = None  # if None, fs would be 22050
        audio_data, sample_rate = librosa.load(file, sr=fs)
        x = audio_data
        fs = sample_rate

        if scale_audio:
            x = x/np.max(np.abs(x))
        f0, voiced_flag = get_pitch(x, fs, winLen=0.02)

        power = np.sum(x**2)/len(x)
        pitch_mean = np.nanmean(f0) if np.mean(np.isnan(f0)) < 1 else 0
        pitch_std = np.nanstd(f0) if np.mean(np.isnan(f0)) < 1 else 0
        voiced_fr = np.mean(voiced_flag)

        stft = np.abs(librosa.stft(audio_data))

        mfcc = np.mean(librosa.feature.mfcc(
            y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        features.extend(mfcc)  # 40 = 40

        chroma = np.mean(librosa.feature.chroma_stft(
            S=stft, sr=sample_rate).T, axis=0)
        features.extend(chroma)  # 12 = 52

        # mel = np.mean(librosa.feature.melspectrogram(
        #     audio_data, sr=sample_rate).T, axis=0)
        # features.extend(mel)  # 128 = 180

        contrast = np.mean(librosa.feature.spectral_contrast(
            S=stft, sr=sample_rate).T, axis=0)
        features.extend(contrast)  # 7 = 187

        # appending four calculated features
        xi = [power, pitch_mean, pitch_std, voiced_fr]
        features.extend(xi)
        # appending all the features
        X.append(features)
        y.append(y1)  # interpretation label
        z.append(y2)  # song label

    return np.array(X), np.array(y), np.array(z)


def scale_data(data, audio_types, labels):
    scaler = StandardScaler()  # define standard scaler

    scaled = scaler.fit_transform(data)  # transform data

    # converting the scaled features into a pandas data frame
    allsong_feature_combine = pd.DataFrame(scaled)
    allsong_feature_combine['audio_type'] = audio_types
    allsong_feature_combine['label'] = labels

    # saving the features into a csv for future reference
    allsong_feature_combine.to_csv(
        'small_dataset/features.csv')
