import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile

def spoof_():
    users = {}
    user_ids = os.listdir("data\\vox1_spoof")
    user_ids = user_ids[0:512]

    fs, audio = wavfile.read('./person8.wav')
    # print(fs)
    ir = []
    names = []
    for filename in os.listdir("data\\vox1_spoof"):
        if 'wav' in filename and 'conv' not in filename:
            # print(filename)
            fs, tmp_ir = wavfile.read('./'+filename)
            tmp_ir = tmp_ir[:500] / (2**15)
            ir.append(tmp_ir)
            names.append(filename)

    ir = np.array(ir)
    plt.figure()
    for it in range(ir.shape[0]):
        """PLOTTING"""
        plt.subplot(ir.shape[0], 2, 2*it+1)
        time_domain = np.linspace(0, len(ir[it, :])/fs, len(ir[it, :]))
        plt.plot(time_domain, ir[it, :])
        plt.xlabel('time (s)')

        """CONVOLVING IMPULSE RESPONSE"""
        convolved_data = np.convolve(audio, ir[it, :], mode='same')
        time_domain = np.linspace(0, len(convolved_data)/fs, len(convolved_data))
        plt.subplot(ir.shape[0], 2, 2*it+2)
        plt.plot(time_domain, convolved_data)

        convolved_data = convolved_data.astype(np.int16)

        wavfile.write('conv_'+names[it], fs, convolved_data)

    plt.show()

