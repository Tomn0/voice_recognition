import librosa
import matplotlib.pyplot as plt
# %%
def display_waveform(y, sr):
    #display waveform (amplitude vs time)
    # import matplotlib.pyplot as plt
    import librosa.display
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y, sr=sr)
    plt.show()

# %%
def display_spectrogram(y, sr):
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
def display_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=24)
    print(mfccs.shape)
    #Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()