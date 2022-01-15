import numpy as np
import librosa.display
import os
from generate_mfccs import mfccs_gen
from generate_mfccs import create_user_sample
import pickle
import time
from datetime import datetime


GLOBAL_PATH = "data\\vox1_dev_wav_partaa_unzip"
irs_paths = ["iPad_ir.wav", "iPhone_ir.wav", "Behritone_ir.wav"]
irs = []

"""Initiate IRs"""
print("#"*20)
print("Initiating IRs...")
for ind, ir_path in enumerate(irs_paths):
    ir, fs_ir = librosa.load(f"irs\\{irs_paths[ind]}", sr=22050)
    irs.append((ir, fs_ir))

print("Done")
print("#"*20)
print('\n'*2)


def convolve_audio(user_audio, which_audio):
    ir, fs_ir = irs[which_audio]

    # ir = np.array(tmp_ir)   # chyba nie jest potrzebne

    """CONVOLVING IMPULSE RESPONSE"""
    convolved_data = np.convolve(user_audio, ir, mode='same')

    return convolved_data


chunks = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260]
# brakuje 10-20


start = time.time()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("#"*20)
print("Starting program:")
print("Current Time =", current_time)
print("#"*20)
print('\n'*2)
for i, chunk in enumerate(chunks):
    loop_start = time.time()

    users = {}
    user_ids = os.listdir(GLOBAL_PATH)
    user_ids = user_ids[chunk:chunk+10]

    for user_id in user_ids:
        user_folder_path = f"{GLOBAL_PATH}\\{user_id}"
        user_sample = create_user_sample(user_folder_path)
        conv = convolve_audio(user_sample, chunk % 3)
        users[f'{user_id}'] = mfccs_gen(user_sample)

    pickle.dump(users, open(f"data\\spoof\\spoof{chunk}-{chunk+10}.p", "wb"))
    loop_end = time.time()
    print("########")
    print(f"End loop: data\\spoof\\spoof{chunk}-{chunk+10}.p")
    print(f"With: irs\\{irs_paths[chunk % 3]}")
    print(f"Elapsed time {loop_end - loop_start} seconds")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print('\n' * 2)

program_end = time.time()
print(program_end - start)
print("#"*20)
print("End program")
print(f"Elapsed time {program_end - start} seconds")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print("#"*20)

