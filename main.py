import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import glob
import os
import wave

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

# librosa.util.example_audio_file()
users = {}
user_ids = os.listdir("data\\vox1_dev_wav_partaa_unzip")
for user_id in user_ids:
    sample = np.array([])
    for file in glob.iglob(f"data\\vox1_dev_wav_partaa_unzip\\{user_id}\\**\\*.wav"):
        y, sr = librosa.load(file)  # if error: add sr=sr
        assert sr == 22050, f"wrong sampling rate for {file} - {sr}"
        duration_in_seconds = float(len(y) / sr)
        sample = np.append(sample, y)
    users[f'{user_id}'] = sample
    
len(users)
# %%
#display waveform (amplitude vs time)
# import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)
plt.show()

# %%
#display Spectrogram
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#If to pring log of frequencies
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

# %%
mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=24)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

# %%
# ogranąc do czego są te wszystkie opcje
mfcc_librosa = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

# %%
print(mfccs.shape)
print(type(mfccs))
