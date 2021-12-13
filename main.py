import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt

# pomoc: https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
# MFCC params
n_mfcc = 39
n_mels = 40
n_fft = 512
hop_length = 160
fmin = 0
fmax = None
sr = 16000

# librosa.util.example_audio_file()
y, sr = librosa.load("data\\vox1_dev_wav_partaa_unzip\\id10001\\1zcIwhmdeo4\\00001.wav", sr=sr)
print(type(y), type(sr))

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
mfccs = librosa.feature.mfcc(y, sr=sr)
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
print(type(mfccs))
