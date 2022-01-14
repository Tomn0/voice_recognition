import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
from generate_mfccs import mfccs_gen
from generate_mfccs import create_user_sample
from util_functions import display_mfccs, display_waveform, display_spectrogram
import pickle


GLOBAL_PATH = "data\\vox1_spoof"


def convolve_audio(user_audio):
    # prepare audio to conv
    # fs_ir, tmp_ir = wavfile.read("iPad_ir.wav")
    tmp_ir, fs_ir = librosa.load("iPad_ir.wav", sr=22050)

    # display_waveform(tmp_ir, fs_ir)
    # display_spectrogram(tmp_ir, fs_ir)

    # tmp_ir, fs_ir = librosa.load("iPad_ir.wav", sr=22050)
    # tmp_ir, fs_ir = sf.read("iPad_ir.wav")
    # """modify wavfile.read output to match output from librosa.load"""
    # nbits = 16
    # tmp_ir /= 2 ** (nbits - 1)
    # tmp_ir = tmp_ir[:500] / (2 ** 15)     # THIS IS THE SAME!

    ir = np.array(tmp_ir)   #chyba nie jest potrzebne

    """CONVOLVING IMPULSE RESPONSE"""
    convolved_data = np.convolve(user_audio, ir, mode='same')
    # convolved_data = convolved_data.astype(np.int16)  # zeruje całą macierz
    # sf.write(f'{GLOBAL_PATH}\\conv_.wav', convolved_data, samplerate=fs_ir)
    # wavfile.write(f'{GLOBAL_PATH}\\conv_.wav', fs_ir, convolved_data)

    return convolved_data


users = {}
user_ids = os.listdir(GLOBAL_PATH)
print(user_ids)

for user_id in user_ids:
    user_folder_path = f"{GLOBAL_PATH}\\{user_id}"
    user_sample = create_user_sample(user_folder_path)
    conv = convolve_audio(user_sample)
    users[f'{user_id}'] = mfccs_gen(user_sample)

    # librosa.display.specshow(users['id10001'], sr=22050, x_axis='time')
    # plt.show()

pickle.dump(users, open("data\\test_spoof_10.p", "wb"))

