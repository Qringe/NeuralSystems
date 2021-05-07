import os.path as path
import numpy as np
from scipy.io import loadmat
from scipy import signal, ndimage
from joblib import dump, load

from utils import download

def extract_syllables(Xs_train, ground_truth_tuples):
    """
    Extracts the parts of the spectogram which contain syllables
    """
    syllables = []

    for on, off, spec_index in ground_truth_tuples:
        syllables.append(Xs_train[spec_index].T[on:off].T)

    # This is necessary as the different arrays in syllables don't have all the same dimension
    temp = np.empty(len(syllables), dtype=object)
    temp[:] = syllables
    return temp

def extract_features(syllables):
    """
    Extracts a broad array of possible features from the spectogram snippets
    """
    def build_features(syllable):
        """
        Takes a single syllable and extracts its features
        """

        # Perform edge detection on the spectogram using sobel filters
        dx = ndimage.sobel(syllable, axis=0)
        dy = ndimage.sobel(syllable, axis=1)
        mag = np.hypot(dx, dy)  
        mag *= 255.0 / np.max(mag)

        # Add some basic features
        features = [
            syllable[0].size,           # Length of syllable
            np.sum(syllable) / \
                syllable[0].size,       # Average sum per column
            np.mean(syllable),          # Average intensity (total)
            np.median(syllable),        # Median intensity (total)
            np.std(syllable),           # Standard deviation of data points (total)
            np.std(np.mean(
                syllable,axis=1)),      # Standard deviation of average per row
            np.max(syllable) - \
                np.min(syllable),       # Range (total)
            np.sum(mag),               # Sum of gradients
            np.mean(mag),              # Average of gradients
            np.std(mag),               # Standard deviation of gradient
        ]

        # Add the same features on a per row basis
        features.extend(np.concatenate((
            np.mean(syllable,axis=1),   # Average intensity (per row)
            np.median(syllable,axis=1), # Median intensity (per row)
            np.std(syllable,axis=1),    # Standard deviation of data points (per row)
            #np.max(syllable,axis=1) - \
            #    np.min(syllable,axis=1),# Range (per row)
            np.mean(mag,axis=1),        # Average intensity on mag (per row)
            np.median(mag,axis=1),      # Median intensity on mag (per row)
            np.std(mag,axis=1),         # Standard deviation of data points on mag (per row)
        )))

        # Compute the mean and median of the first 18 columns
        # This helps to distinguish the classes 'B' and 'C'
        features.extend(np.concatenate((
            np.mean(syllable.T[:18].T,axis=1),   # Average intensity (per row)
            np.median(syllable.T[:18].T,axis=1) # Median intensity (per row)
        )))

        return features
    
    return np.array([build_features(syllable) for syllable in syllables])


def create_features():
    # If the training data is not yet available, download it
    base = path.dirname(path.abspath(__file__))
    download()

    # Load the training data
    Xs_train = loadmat(path.join(base,"Data/train_dataset/spectrograms/g17y2_train.mat"))["Xs_train"][0]
    annotations = loadmat(path.join(base,"Data/train_dataset/annotations/annotations_g17y2_train.mat"))["annotations_train"][0]

    # Extract the important data structures from the training data
    SETindex = annotations["y"][0]["SETindex"][0][0][0]-1
    ons = annotations["y"][0]["onset_columns"][0][0][0]-1
    offs = annotations["y"][0]["offset_columns"][0][0][0]-1
    ground_truth_tuples = [(a,b,c) for a,b,c in zip(ons,offs,SETindex)]

    # Extract the features
    data_path = path.join(base,"Data/Xs_features.data")
    if not path.isfile(data_path):
        syllables = extract_syllables(Xs_train, ground_truth_tuples)
        Xs_features = extract_features(syllables)
        dump(Xs_features, data_path) 
    else:
        Xs_features = load(data_path)
    
if __name__ == "__main__":
    create_features()