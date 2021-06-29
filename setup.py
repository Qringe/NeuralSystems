import os
from os import path
from utils import DATA_PATH, BIRD_DATA_PATH, download

if __name__ == "__main__":
    # Downloads the training data, if it is not already present
    download()

    # Checks if the Bird_Data folder exists. If not, create it
    base = path.dirname(path.abspath(__file__))
    p = path.join(base, BIRD_DATA_PATH)
    if not path.isdir(p):
        os.system(f"mkdir {p}")
