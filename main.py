import numpy as np
import librosa
import librosa.display
import glob
import os
import pickle
import matplotlib.pylab as plt
from util_functions import display_mfccs, display_waveform, display_spectrogram


# pomoc: https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
# MFCC params
# dokładniejszy opis poszczególnych parametrów:
# https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature
# n_mfcc = 24
# n_mels = 40
# n_fft = 512
# hop_length = 160
# fmin = 0
# fmax = None
# sr = 16000
# # ogranąc do czego są te wszystkie opcje
# mfcc_librosa = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft,
#                                     n_mfcc=n_mfcc, n_mels=n_mels,
#                                     hop_length=hop_length,
#                                     fmin=fmin, fmax=fmax, htk=False)
# mfcc global params
n_ftt = 512
hop_length = 512


def main():

    # librosa.util.example_audio_file()
    users = {}
    user_ids = os.listdir("data\\test")
    user_ids = user_ids[0:512]
    for user_id in user_ids:
        sample = np.array([])
        for file in glob.iglob(f"data\\test\\{user_id}\\**\\*.wav"):
            y, sr = librosa.load(file)  # if error: add sr=sr
            assert sr == 22050, f"wrong sampling rate for {file} - {sr}"
            sample = np.append(sample, y)
            # 200 * 22050
            duration_in_seconds = float(len(sample) / sr)
            if duration_in_seconds > 200:
                sample = sample[:4410000]
                break
        sr = 22050
        mfcc = librosa.feature.mfcc(sample, sr=sr, n_mfcc=13)
        mfcc = mfcc[1:]     # pierwsza cecha to moc - nie jest nam potrzebna (czy na pewno?)

        delta_mfcc = librosa.feature.delta(mfcc, axis=1)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, axis=1)
        mfccs = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)

        users[f'{user_id}'] = mfccs
    len(users)
    pickle.dump(users, open("data/test.p", "wb"))


main()

