import os

from generate_mfccs import mfccs_gen

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if "users512.p" not in os.listdir(dir_path + "data"):
        mfccs_gen()

main()

