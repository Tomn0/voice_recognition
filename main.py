import numpy as np

from os import listdir
from os.path import isfile, join
# list files in directory
files = [f for f in listdir("data") if isfile(join("data", f))]

# with open("data\\vox1_dev_wav_partaa_unzip\\id10001\\1zcIwhmdeo4\\00001.wav") as f: