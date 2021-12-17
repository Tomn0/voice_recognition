"""This module contains utility functions to display plots of analyzed audio data"""
import librosa.display
import matplotlib.pyplot as plt


# %%
def display_waveform(y, sr):
    """
    display waveform (amplitude vs time)
    :param y: audio time series
    :param sr: sampling rate
    :return: None
    """
    # import matplotlib.pyplot as plt
    import librosa.display
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y, sr=sr)
    plt.show()


# %%
def display_spectrogram(y, sr):
    """
    displays spectrogram - spectrum of frequencies of sound as they vary with time
    :param y: audio time series
    :param sr: sampling rate
    :return: None
    """
    # display Spectrogram
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    # If to pring log of frequencies
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()


# %%
def display_mfccs(y, sr):
    """
    displays mfccs (Mel-frequency cepstral coefficients)
    :param y: audio time series
    :param sr: sampling rate
    :return: None
    """
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=24)
    # Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()

