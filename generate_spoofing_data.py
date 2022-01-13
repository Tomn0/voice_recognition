import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
from generate_mfccs import mfccs_gen
from generate_mfccs import create_user_sample

# prepare audio to conv
fs_ir, tmp_ir = wavfile.read(open("iPad_ir.wav", 'rb'))
tmp_ir = tmp_ir[:500] / (2 ** 15)

# users = {}
# user_ids = os.listdir("data\\test")
# user_ids = user_ids[0:512]
for filename in os.listdir(f"data\\vox1_spoof\\id10001\\7gWzIy6yIIk"):
    if 'ir' not in filename and 'wav' in filename and 'conv' not in filename:

        fs, audio = wavfile.read("data\\vox1_spoof\\id10001\\7gWzIy6yIIk\\" + filename)

        ir = np.array(tmp_ir)

        """CONVOLVING IMPULSE RESPONSE"""
        convolved_data = np.convolve(audio, ir, mode='valid')
        convolved_data = convolved_data.astype(np.int16)
        # wavfile.write('conv_'+filename, fs_ir, convolved_data)
