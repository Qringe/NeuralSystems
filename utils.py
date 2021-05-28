import os.path as path
import os
import numpy as np
import matplotlib
from scipy.io import loadmat
from scipy.interpolate import interp2d
from joblib import dump, load
from hashlib import sha256
import random

import torch
from torch.utils.data import TensorDataset, DataLoader

##########################################################################################################################
# Overview of this file:
# Global parameters
# General functions
# Data-handling functions

##########################################################################################################################
# Global parameters

# Check if GPU is available
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# The list of available bird data
BIRD_NAMES = ["g17y2", "g19o10", "g4p5", "R3428"]

# The data path and model path
DATA_PATH = "Data/"
DATASET_PATH = DATA_PATH + "Datasets/"
BIRD_DATA_PATH = DATA_PATH + "Bird_Data/"
MODEL_PATH = "Models/"
PREDICTIONS_PATH = DATA_PATH + "Predictions/"

SAMPLING_RATE = 32000

##########################################################################################################################
# General functions


def tuples2vector(meta_tuples):
    """
    Takes as input a list of tuples of the form:
        (
            [ (on1, off1, spec), (on2, off2, spec), ... ],    # A list of tuples denoting syllable positions
            length  # Length of the spectogram from which the above syllables are taken from
        )
    The list is then converted to a long array of 0s and 1s. Example:
        Input: A list containing a single tuple:
        [(
            [ (2, 4, 1), (7, 11, 1), (13, 15, 1)],
            17
        )]
        Output: The tuples converted to a vector:
            [0,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0]
    """

    output_vec = []

    for tup in meta_tuples:
        # Extract the syllables and create an output vector for this spectogram
        tuples, length = tup
        temp_vec = np.zeros(length, dtype=np.int8)

        # Check that tuples are all from the same spectogram
        spec_ids = set([t[2] for t in tuples])
        if len(spec_ids) > 1:
            raise Exception(f"All tuples in a list have to come from the same spectogram! Got ids {spec_ids}")

        # Iterate through all syllables and set the corresponding area in 'temp_vec' to 1
        for syllable in tuples:
            on, off, idx = syllable[:3]
            temp_vec[on:off+1] = 1

        # Append the output vector with the output from this spectogram
        output_vec.extend(temp_vec.tolist())

    return output_vec


def score_predictions(y_true, y_pred, tolerance=4):
    """
    The function used to score the predictions for a spectogram
    """
    t_onset = list(map(lambda tup: tup[0], y_true))
    t_offset = list(map(lambda tup: tup[1], y_true))
    p_onset = list(map(lambda tup: tup[0], y_pred))
    p_offset = list(map(lambda tup: tup[1], y_pred))

    # If there are tuples from multiple spectograms, throw an error
    spec_ids_true = set(map(lambda tup: tup[2], y_true))
    spec_ids_pred = set(map(lambda tup: tup[2], y_pred))

    if len(spec_ids_pred) > 1 or len(spec_ids_true) > 1:
        print("The true labels contain the spectograms = ", spec_ids_true)
        print("The predicted labels contain the spectograms = ", spec_ids_pred)
        raise Exception(
            "The function 'score_predictions' can only score tuples from a single spectogram." +
            "You provided tuples from multiple spectograms!")

    TP = 0
    FP = 0
    tmp = 0
    p_ob_pon, p_ob_poff, p_ob_non, p_ob_noff = [], [], [], []  # just for observing the results
    for i in range(len(p_onset)):
        index = np.zeros(2 * tolerance + 1)
        for j in range(-tolerance, tolerance + 1):
            if p_onset[i] + j in t_onset:
                index[j] = t_onset.index(p_onset[i] + j)
            else:
                index[j] = -1
        tmpidx = list(filter(lambda x: x > -1, index))
        if len(tmpidx) == 0:
            FP = FP + 1
            p_ob_non.append(p_onset[i])
            p_ob_noff.append(p_offset[i])
            tmp = -1
        else:
            for k in range(len(tmpidx)):
                tmpidx[k] = int(tmpidx[k])
                if abs(t_offset[tmpidx[k]] - p_offset[i]) <= tolerance:
                    TP = TP + 1
                    tmp = tmp + 1
                    p_ob_pon.append(p_onset[i])
                    p_ob_poff.append(p_offset[i])
        if tmp > 1:
            print("You need to see what happens")
        if tmp == 0:
            FP = FP + 1
            p_ob_non.append(p_onset[i])
            p_ob_noff.append(p_offset[i])
        else:
            tmp = 0

    precision = TP / (FP + TP + 1e-12)
    recall = TP / len(t_onset)
    acc_rate = 2 * precision * recall / (precision + recall + 1e-12)

    return acc_rate


def hash_spectograms(spectograms, ndigits=6):
    """
    Just a small deterministic hashfunction. This function is used to identify files.
    The parameter 'ndigits' denotes the amount of digits of the hash which shall be returned
    """
    h = ""
    for spec in spectograms:
        h += sha256(spec).hexdigest()
    return sha256(h.encode('utf-8')).hexdigest()[-ndigits:]


def reverse_on_off(tuples, T):
    reversed_tuples = []
    next_on = 0
    if tuples == []:
        return []
    for on, off, idx in tuples:
        reversed_tuples.append((next_on, on, idx))
        next_on = off
    reversed_tuples.append((next_on, T-1, idx))
    reversed_tuples = [(a, b, c) for (a, b, c) in reversed_tuples if not a == b]
    return reversed_tuples


def extract_features(X):
    X = np.array(X, dtype=np.float32)
    return [np.mean(X), np.std(X), np.median(X), np.mean(np.abs(X)), np.max(X)-np.min(X)]


def plot_spectogram(ax, X, tuples_predicted, tuples_gt):
    ax.imshow(np.flip(X, axis=0))
    for t_p in tuples_predicted:
        ax.add_patch(matplotlib.patches.Rectangle((t_p[0], 0), t_p[1]-t_p[0], 15, linewidth=2,
                                                  edgecolor='r',
                                                  facecolor='r',
                                                  fill=True)
                     )
    for t_gt in tuples_gt:
        ax.add_patch(matplotlib.patches.Rectangle((t_gt[0], 30), t_gt[1]-t_gt[0], 15, linewidth=2,
                                                  edgecolor='g',
                                                  facecolor='g',
                                                  fill=True)
                     )


def normalize(X, mean=None, std=None):
    if not (mean is None):
        return (X - mean) / std
    else:
        return (X - torch.mean(X)) / torch.std(X)

##########################################################################################################################
# Data-handling functions


def download():
    """
    Checks if the data files are present and otherwise downloads them
    """
    base = path.dirname(path.abspath(__file__))
    p = path.join(base, DATA_PATH + "train_dataset_new.zip")

    # Remove the old file and directory if it is still present
    if path.isfile(path.join(base, DATA_PATH + "train_dataset.zip")):
        print("Removing old data")
        os.system("rm " + DATA_PATH + "train_dataset.zip")
        os.system("rm -r " + DATA_PATH + "train_dataset")

    # If the data doesn't exist yet, download the zip
    if not path.isfile(p):
        # pip install gdown
        os.system("gdown https://drive.google.com/uc?id=1jQTe1L2UKtnuTiE-768EfhhFW5O8_p33")
        os.system("mv train_dataset.zip " + DATA_PATH + "train_dataset_new.zip")
        os.system("unzip " + DATA_PATH + "train_dataset_new.zip -d " + DATA_PATH)


def load_bird_data(names=None, indices=None, drop_unlabelled=False, resample=SAMPLING_RATE):
    """
    Loads the data of one or several birds. The parameter 'names' can be either a list
    containing a subset from the following list ["g17y2", "g19o10", "g4p5", "R3428"], or
    just a single name from the list above, in case you would only like to load the data
    of one bird.

    Alternatively, you can also use indices instead of names. The mapping of indices is as follows:
        0 -> g17y2
        1 -> g19o10
        2 -> g4p5
        3 -> R3428
    As for 'names' you can specify this parameter either as a list or a single integer

    The parameter 'names' has priority over the parameter 'indices'. If you specify both
    parameters, then only 'names' will be read.

    The parameter 'resample' can either have the boolean value 'False', indicating that no spectograms
    should be resampled, or an integer value, in which case the spectograms get resampled to the sample rate
    indicated by 'resample'

    If you don't specify any parameters, the data of all birds will be returned.
    """

    bird_names_np = np.array(BIRD_NAMES)

    # Choose from 'bird_names_np' all birds whose data shall be loaded
    if names is not None:
        if type(names) != list:
            birds = [names]
        else:
            birds = names
    elif indices is not None:
        birds = bird_names_np[indices]
    else:
        birds = bird_names_np

    base = path.dirname(path.abspath(__file__))

    bird_data = {}

    # Iterate through all selected birds
    for bird in birds:
        Xs_train = loadmat(path.join(base, DATA_PATH + f"train_dataset/spectrograms/{bird}_train.mat"))["Xs_train"][0]
        annotations = loadmat(
            path.join(base, DATA_PATH + f"train_dataset/annotations/annotations_{bird}_train.mat"))["annotations_train"][0]

        SETindex = annotations["y"][0]["SETindex"][0][0][0]-1
        scanrate = annotations['scanrate'][0][0][0]
        if ('SETindex_labelled', 'O') in eval(str(annotations.dtype)):
            labelled_specs = annotations['SETindex_labelled'][0][0] - 1
        else:
            labelled_specs = list(range(len(Xs_train)))
        ons = annotations["y"][0]["onset_columns"][0][0][0]-1
        offs = annotations["y"][0]["offset_columns"][0][0][0]-1
        ground_truth_tuples = [(a, b, c, bird) for a, b, c in zip(ons, offs, SETindex)]

        # NOTE: The 'order = C' is required to make the hash function 'hash_spectograms' work properly!
        for i in range(len(Xs_train)):
            Xs_train[i] = Xs_train[i].copy(order='C')
            Xs_train[i] = Xs_train[i].astype(np.float)

        current_data = {
            "Xs_train": Xs_train,
            "tuples": ground_truth_tuples,
            "ids_labelled": labelled_specs
        }

        # The data can be either accessed using the string name of the bird, or the index
        bird_data[bird] = current_data

        # If the scanrate differes from 'SAMPLING_RATE' and the parameter 'resample' is not 'False'
        # resample the data to 'SAMPLING_RATE'
        if resample is not False and type(resample) == int and scanrate != SAMPLING_RATE:
            bird_data = resample_data(bird_data, bird_names=[bird], old_frequencies=scanrate,
                                      new_frequencies=SAMPLING_RATE, name="on_load")

    if drop_unlabelled:
        bird_data, unlabelled_data = extract_labelled_spectograms(bird_data)

    return bird_data


def extract_labelled_spectograms(bird_data):
    """
    Takes as input a dictionary containing the data of different birds (as returned by the function 'load_bird_data') and
    splits the data of each bird into a set of spectograms which are labelled and a set of spectograms which are unlabelled.

    'bird_data' should have the form:
        {
            "bird_name1": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples, "ids_labelled" : ids_labelled},
            "bird_name2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples, "ids_labelled" : ids_labelled},
            ...
        }
    where "Xs_train" is a list of spectograms and "tuples" a list of tuples of the form (on,off,SETindex,bird_name) denoting
    the start (on) and end (off) of a syllable, and the spectogram index (SETindex) of said syllable, and the name of the bird
    (bird_name) from which the syllable stems from. Finally, "ids_labelled" is a list of spectograms which are actually labelled.
    """
    bird_data_labelled = {}
    bird_data_unlabelled = {}

    # Iterate over each bird in 'bird_data'
    for bird_name in bird_data.keys():
        bird_data_labelled[bird_name] = {}
        bird_data_unlabelled[bird_name] = {}

        # Load the data of each bird
        Xs_train_whole = bird_data[bird_name]["Xs_train"]
        tuples_whole = bird_data[bird_name]["tuples"]
        ids_labelled = bird_data[bird_name]["ids_labelled"]

        # Get the ids of all unlabelled spectograms
        ids_unlabelled = np.delete(np.array(range(len(Xs_train_whole))), ids_labelled)

        # A dictionary which maps the old ids of 'ids_labelled' to a contiguous range 0-len(ids_labelled)
        label_mapper = {}
        new_index = 0
        for label in ids_labelled:
            label_mapper[label] = new_index
            new_index += 1

        # Extract all labelled data
        bird_data_labelled[bird_name]["Xs_train"] = Xs_train_whole[ids_labelled]
        tuples_reduced = list(map(lambda tup: (tup[0], tup[1], label_mapper[tup[2]], tup[3]), tuples_whole))
        bird_data_labelled[bird_name]["tuples"] = tuples_reduced

        # Extract all unlabelled data
        bird_data_unlabelled[bird_name]["Xs_train"] = Xs_train_whole[ids_unlabelled]
        bird_data_unlabelled[bird_name]["tuples"] = []

    return bird_data_labelled, bird_data_unlabelled


def train_test_split(bird_data, configs=None, seed=None):
    """
    Takes as input a dictionary containing the data of different birds (as returned by the function
    'extract_labelled_spectograms') and uses the configuration dictionary 'configs' to split each of the items of the
    'bird_data' dict into a training and test set.

    'bird_data' should have the form:
        {
            "bird_name1": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
            "bird_name2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
            ...
        }
    where "Xs_train" is a list of spectograms and "tuples" a list of tuples of the form (on,off,SETindex,bird_name) denoting
    the start (on) and end (off) of a syllable, and the spectogram index (SETindex) of said syllable, and the name of the bird
    (bird_name) from which the syllable stems from.

    'configs' can be either (a) an integer, (b) a float, or (c) a dictionary of ints/floats.
        (a) If 'configs' is an integer 'i', then the function will put exactly 'i' random spectograms per bird in the test set.
        (b) If 'configs' is a float 0 <= f <= 1, then this function will put for each bird exactly a fraction 'f' of all
            spectograms into the test set.
        (c) If 'configs' is a dictionary it should have the form:
                {
                    "bird_name1": freq1 or tot1,
                    "bird_name2": freq2 or tot1,
                    ...
                }
            where the keys are the bird names and each key either contains an int 'tot' denoting the total number of windows
            which will be put into the test set for this bird, or a float 0 <= f <= 1 denoting the fraction of windows which
            will be put into the test set for this bird
    """
    bird_data_train = {}
    bird_data_test = {}

    if seed is not None and type(seed) == int:
        random.seed(seed)

    # Handle configs
    if configs is None:
        configs = {bird_name: 0.2 for bird_name in bird_data.keys()}
    elif type(configs) == int or type(configs) == float:
        configs = {bird_name: configs for bird_name in bird_data.keys()}
    elif type(configs) == dict:
        pass
    else:
        raise Exception("The variable 'configs' has the wrong format! " +
                        "Check out the function docstring for an explanation of the format.")

    # Iterate over each bird in 'bird_data'
    for bird_name in bird_data.keys():
        bird_data_train[bird_name] = {}
        bird_data_test[bird_name] = {}

        # Load the data of each bird
        Xs_train_whole = bird_data[bird_name]["Xs_train"]
        tuples_whole = bird_data[bird_name]["tuples"]
        test_freq = configs[bird_name]

        # Get the ids of the test set
        ids = list(range(len(Xs_train_whole)))
        if test_freq >= 0 and test_freq <= 1:
            test_ids = random.sample(ids, round(test_freq * len(ids)))
        elif test_freq > 1:
            if test_freq > len(ids):
                test_freq = len(ids)
            test_ids = random.sample(ids, round(test_freq))
        else:
            raise Exception(f"'test_freq' needs to be non-negative! You provided {test_freq}")

        # Get the ids of the train set
        train_ids = np.delete(ids, test_ids)

        # A dictionary which maps the old ids of 'id_list' to a contiguous range 0-len(id_list)
        def build_mapper(id_list):
            label_mapper = {}
            new_index = 0
            for label in id_list:
                label_mapper[label] = new_index
                new_index += 1
            return label_mapper

        # Extract the train data
        bird_data_train[bird_name]["Xs_train"] = Xs_train_whole[train_ids]
        train_mapper = build_mapper(train_ids)
        tuples_train = [t for t in tuples_whole if t[2] in train_ids]
        tuples_train = list(map(lambda tup: (tup[0], tup[1], train_mapper[tup[2]], tup[3]), tuples_train))
        bird_data_train[bird_name]["tuples"] = tuples_train

        # Extract the test data
        bird_data_test[bird_name]["Xs_train"] = Xs_train_whole[test_ids]
        test_mapper = build_mapper(test_ids)
        tuples_test = [t for t in tuples_whole if t[2] in test_ids]
        tuples_test = list(map(lambda tup: (tup[0], tup[1], test_mapper[tup[2]], tup[3]), tuples_test))
        bird_data_test[bird_name]["tuples"] = tuples_test

    return bird_data_train, bird_data_test


def resample_data(bird_data, bird_names, old_frequencies, new_frequencies, name, kind="cubic", read_cache=True, write_cache=True):
    """
    Takes as input a dictionary 'bird_data' containing the data of different birds (as returned by the function 'load_bird_data'),
    and a list of names 'bird_names'. The data of the birds specified in this list gets then resampled.

    'old_frequencies' can be either an integer or a dictionary. If it is an integer, it represents the old sampling frequency of
    all birds specified in 'bird_names'. If different birds in 'bird_data' have different old sampling frequencies, you can
    specify the different sampling frequencies as follows:
        {
            "bird_name1": old_frequency1
            "bird_name2": old_frequency2
            ...
        }

    'new_frequencies' can be either an integer or a dictionary. If it is an integer, it represents the new sampling frequency of
    all birds specified in 'bird_names'. If different birds in 'bird_data' shall have different sampling frequencies, you can
    specify the different sampling frequencies as follows:
        {
            "bird_name1": new_frequency1
            "bird_name2": new_frequency2
            ...
        }

    'name' is used to check whether the spectogram data has already been resampled once.

    'kind' specifies how the 2d interpolation shall be done. This function performs spline interpolation. The parameter 'kind'
    can set the degree of the splines. Possible values are: ["linear", "cubic", "quintic"]

    If 'read_cache == True', then the
    function first checks, whether the spectograms have already been resampled once (i.e.
    whether a file with name 'name' already exists) and if so, loads the data.

    If 'write_cache==True' then the function will store the computed results.
    """
    base = path.dirname(path.abspath(__file__))

    # Convert the 'old_frequencies' input to a dictionary if it isn't already a dictionary.
    if type(old_frequencies) != dict:
        freq = old_frequencies
        old_frequencies = {bird_name: freq for bird_name in bird_names}

    # Convert the 'new_frequencies' input to a dictionary if it isn't already a dictionary.
    if type(new_frequencies) != dict:
        freq = new_frequencies
        new_frequencies = {bird_name: freq for bird_name in bird_names}

    # Iterate over all birds whose data should be resampled
    for bird_name in bird_names:
        spectograms = bird_data[bird_name]["Xs_train"]
        tuples = bird_data[bird_name]["tuples"]

        # Check if the data has already been resampled once and if so, load it
        storage_path = path.join(base, DATASET_PATH + f"resampled_{name}_{bird_name}_hash_{hash_spectograms(spectograms)}.data")
        if read_cache and path.isfile(storage_path):
            print("Load cached resampled data of bird ", bird_name)
            bird_data[bird_name] = load(storage_path)
            continue

        print("Resampling spectograms of bird ", bird_name)

        old_freq = old_frequencies[bird_name]
        new_freq = new_frequencies[bird_name]

        new_spectograms = []
        new_tuples = []

        # Iterate over all spectograms of this bird
        for index, spectogram in enumerate(spectograms):
            if index % 20 == 0:
                print(index, " / ", len(spectograms))

            # Compute the length of the spectogram in seconds
            rows, cols_old = spectogram.shape
            seconds = rows * cols_old / old_freq

            # Compute the dimensions of the new spectogram
            cols_new = round(seconds * new_freq / rows)

            # Compute for each column the corresponding time index
            old_time_stamps = np.linspace(0, seconds, cols_old)

            # Create the timestamps for the resampled spectogram
            new_time_stamps = np.linspace(0, seconds, cols_new)

            # Build an interpolation function using the old timestamps and data
            func = interp2d(x=old_time_stamps, y=list(range(rows)), z=spectogram, kind=kind, copy=False)

            # Get the resampled spectogram values using the interpolation function 'func'
            resampled_spectogram = func(x=new_time_stamps, y=list(range(rows)))

            new_spectograms.append(resampled_spectogram)

            # Adjust the start and end of the syllable tuples
            current_tuples = [tup for tup in tuples if tup[2] == index]

            # A function which adjusts the on- and off indices of the tuple and leaves the remaining
            # entries untouched
            def resample_tuple(tup, cols_old, cols_new):
                on = round(tup[0] * (cols_new / cols_old))
                off = round(tup[1] * (cols_new / cols_old))
                return (on, off, *tup[2:])

            # Apply this function to all tuples
            resampled_tuples = [resample_tuple(tup, cols_old, cols_new) for tup in current_tuples]

            new_tuples.extend(resampled_tuples)

        # This is ugly but necessary because the different spectograms have different lengths, so
        # np.array(new_spectograms) doesn't work
        for i in range(len(new_spectograms)):
            bird_data[bird_name]["Xs_train"][i] = new_spectograms[i].copy(order='C')
        bird_data[bird_name]["tuples"] = new_tuples

        # Store the results, if write_cache = True
        if write_cache:
            dump(bird_data[bird_name], storage_path)

    return bird_data


def standardize_data(bird_data, coarse_mode="per_spectogram", fine_mode="per_row"):
    """
    Takes as input a dictionary containing the data of different birds (as returned by the function 'load_bird_data') and
    standardizes the spectograms. The standardization can be either done for each spectogram (coarse_mode = "per_spectogram")
    or for each bird (coarse_mode = "per_bird"). Furthermore, you can decide, whether the standardization should be done
    per row (fine_mode = "per_row"), in which case the means and stds are computed per row, or if you want to use just one scalar
    for the mean and std (fine_mode = "scalar"). The two parameters 'coarse_mode' and 'fine_mode' can be combined arbitrarily.

    'bird_data' should have the form:
        {
            "bird_name1": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples, "ids_labelled" : ids_labelled},
            "bird_name2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples, "ids_labelled" : ids_labelled},
            ...
        }
    """
    if coarse_mode not in ["per_bird", "per_spectogram"]:
        raise Exception(f"'coarse_mode' must be one of ['per_bird', 'per_spectogram'], but you provided {coarse_mode}")
    if fine_mode not in ["per_row", "scalar"]:
        raise Exception(f"'fine_mode' must be one of ['per_row', 'scalar'], but you provided {fine_mode}")

    converted = bird_data.copy()

    # Iterate over all birds
    for bird_name in bird_data.keys():
        specs = bird_data[bird_name]["Xs_train"]

        if coarse_mode == "per_bird":
            temp = np.concatenate(specs, axis=1)
            if fine_mode == "scalar":
                means = np.mean(temp)
                stds = np.std(temp)
            elif fine_mode == "per_row":
                means = np.mean(temp, axis=1).reshape((specs[0].shape[0], -1))
                stds = np.std(temp, axis=1).reshape((specs[0].shape[0], -1))

            for i in range(specs.shape[0]):
                specs[i] = (specs[i] - means) / stds

        elif coarse_mode == "per_spectogram":
            for i in range(specs.shape[0]):
                if fine_mode == "scalar":
                    specs[i] = (specs[i] - np.mean(specs[i])) / np.std(specs[i])
                elif fine_mode == "per_row":
                    means = np.mean(specs[i], axis=1).reshape((specs[i].shape[0], -1))
                    stds = np.std(specs[i], axis=1).reshape((specs[i].shape[0], -1))
                    specs[i] = (specs[i] - means) / stds

        converted[bird_name]["Xs_train"] = specs

    return converted


def extract_birds(bird_data, bird_names):
    """
    A small convenience function which extracts from the 'bird_data' dictionary the birds whose
    names are stored in 'bird_names'
    """
    bird_data_new = {}

    # Ensure that bird_names is actually a list
    if type(bird_names) != list:
        bird_names = [bird_names]

    # Copy the birds which are specified in bird_names
    for bird_name in bird_names:
        if bird_name not in bird_data.keys():
            print(
                f"Warning: The data of bird '{bird_name}' seems to be missing in the provided bird data. " +
                "It is therefore skipped.")
            continue
        bird_data_new[bird_name] = bird_data[bird_name]

    return bird_data_new


def compute_max_window_sizes(bird_data, wnd_sz, dt):
    """
    A small helper function which computes for each bird in bird_data the maximum possible number
    of windows which can be extracted when using the window size 'wnd_sz' and the stride 'dt'.
    """
    result = {}
    for bird_name in bird_data.keys():
        spectograms = bird_data[bird_name]["Xs_train"]
        total_amount = sum(map(lambda spec: len(range(0, spec.shape[1]-wnd_sz, dt)), spectograms))
        result[bird_name] = total_amount

    return result


def extract_from_spectogram(spectogram, tuples, wnd_sz, dt, feature_extraction=False, online=False):
    """
    Takes a single spectogram and the corresponding tuples. The function then "walks" over the spectogram
    from left to right and extracts windows of size 'wnd_sz', using a stride of 'dt'. It keeps track of the
    amount of on- and off-tuples and finally returns the resulting windows in two separate arrays.
    """
    # Array 'X_on' stores the extracted on-windows and 'X_off' the windows of type off
    X_on = []
    X_off = []

    # A helper function which extracts a window at a given position (if possible)
    def get_wnd_data(t):
        if online:
            left = int(t-wnd_sz)
            right = int(t)
        else:
            left = int(t-wnd_sz)
            right = int(t+wnd_sz)
        if left < 0 or right >= spectogram.shape[1]:
            return None
        if feature_extraction:
            X_tmp = extract_features(spectogram[:, left:right])
        else:
            X_tmp = spectogram[:, left:right]
        return X_tmp

    # "Walk" over the spectogram from left to right, making strides of size 'dt'
    for t in range(0, spectogram.shape[1], dt):
        X_tmp = get_wnd_data(t)
        if isinstance(X_tmp, type(None)):
            continue
        window_is_on = any(map(lambda tup: tup[0] <= t and tup[1] >= t, tuples))
        if window_is_on:
            X_on.append(X_tmp)
        else:
            X_off.append(X_tmp)

    return X_on, X_off


def create_windows(bird_data, wnd_sizes, feature_extraction=False, limits=None,  on_fracs=None, dt=5, online=False, seed=None):
    """
    For every window size in 'wnd_sizes' convert the spectograms in the 'bird_data' dictionary to
    a set of windows.

    'bird_data' should have the form:
        {
            "bird_name1": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
            "bird_name2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
            ...
        }
    where "Xs_train" is a list of spectograms and "tuples" a list of tuples of the form (on,off,SETindex,bird_name) denoting
    the start (on) and end (off) of a syllable, and the spectogram index (SETindex) of said syllable, and the name of the bird
    (bird_name) from which the syllable stems from.

    'wnd_sizes' can be either an integer or a list of integers denoting different window sizes.

    'limits' can be either (a) an integer, (b) a float, or (c) a dictionary of ints/floats.
        (a) If 'limits' is an integer 'i', then this function will return exactly 'i' windows per bird.
        (b) If 'limits' is a float 0 <= f <= 1, then this function will return for each bird exactly the fraction 'f' of all
            possible windows.
        (c) If 'limits' is a dictionary it should have the form:
                {
                    "bird_name1": freq1 or tot1,
                    "bird_name2": freq2 or tot1,
                    ...
                }
            where the keys are the bird names and each key either contains an int 'tot' denoting the total number of windows
            which will be returned for this bird, or a float 0 <= f <= 1 denoting the fraction of windows which will be
            returned for this bird

    'on_fracs' can be either (a) an integer, (b) a float, or (c) a dictionary of ints/floats.
        (a) If 'on_fracs' is an integer 'i', then this function will return exactly 'i' on-windows per bird and the rest of
        windows will all be off-windows.
        (b) If 'on_fracs' is a float 0 <= f <= 1, then for each bird, a fraction 'f' of the returned windows will be on-windows
        while the rest will be off-windows.
        (c) If 'on_fracs' is a dictionary it should have the form:
                {
                    "bird_name1": freq1 or tot1,
                    "bird_name2": freq2 or tot1,
                    ...
                }
            where the keys are the bird names and each key either contains an int 'tot' denoting the total number of windows
            which will be on-windows for this bird, or a float 0 <= f <= 1 denoting the fraction of windows which will be
            on-windows for this bird.

    'dt' is the stride with which the windows should be sampled.

    'online' is a boolean variable indicating whether only spectogram columns before the target column should be extracted or
    also columns after the target column
    """
    # Set seed if necessary
    if seed is not None and type(seed) == int:
        random.seed(seed)

    # Handle the 'wnd_sizes' variable
    if type(wnd_sizes) == int:
        wnd_sizes = [wnd_sizes]

    # Handle the 'limits' input
    # If 'limits' is not specified, return the same amount of windows per bird
    if limits is None:
        limits = {bird_name: 20000 for bird_name in bird_data.keys()}
    elif type(limits) == int or type(limits) == float:
        limits = {bird_name: limits for bird_name in bird_data.keys()}
    elif type(limits) == dict:
        pass
    else:
        raise Exception("The variable 'limits' has the wrong format! " +
                        "Check out the function docstring for an explanation of the format.")

    # Handle the 'on_fracs' input
    # If 'on_fracs' is not specified, return for each bird 50% on-windows and 50% off-windows
    if on_fracs is None:
        on_fracs = {bird_name: 0.5 for bird_name in bird_data.keys()}
    elif type(on_fracs) == int or type(on_fracs) == float:
        on_fracs = {bird_name: on_fracs for bird_name in bird_data.keys()}
    elif type(on_fracs) == dict:
        pass
    else:
        raise Exception("The variable 'on_fracs' has the wrong format! " +
                        "Check out the function docstring for an explanation of the format.")

    # The final results dictionary
    results = {}
    real_configs = {}

    # Iterate over all window sizes and create the windows
    for wnd_sz in wnd_sizes:
        # For each bird, compute the maximum possible number of windows:
        max_sizes = compute_max_window_sizes(bird_data, wnd_sz, dt)

        results[wnd_sz] = {}
        real_configs[wnd_sz] = {}

        # Iterate over each bird in 'bird_data'
        for bird_name in bird_data.keys():
            results[wnd_sz][bird_name] = {}
            real_configs[wnd_sz][bird_name] = {}

            spectograms = bird_data[bird_name]["Xs_train"]
            tuples = bird_data[bird_name]["tuples"]

            # If 'limits' contains fractions, compute the corresponding absolute numbers
            if type(limits[bird_name]) == float:
                limit = round(max_sizes[bird_name] * limits[bird_name])
            else:
                limit = limits[bird_name]

            # If 'on_fracs' contains fractions, compute the corresponding absolute numbers
            if on_fracs[bird_name] > limit:
                on_num = limit
            elif type(on_fracs[bird_name]) == float:
                on_num = round(limit * on_fracs[bird_name])
            else:
                on_num = on_fracs[bird_name]
            off_num = limit - on_num

            # Get a shuffled list of the indices of the spectograms
            ids = random.sample(range(len(spectograms)), len(spectograms))

            # Two variables denoting how many spectograms on- and off-windows have already
            # been sampled
            curr_on = curr_off = 0
            on_windows = []
            off_windows = []

            # Iterate over all spectograms of this bird
            for id in ids:
                spectogram = spectograms[id]
                spec_tuples = [t for t in tuples if t[2] == id]

                # Extract the windows of this spectogram
                on_windows_tmp, off_windows_tmp = extract_from_spectogram(
                    spectogram=spectogram,
                    tuples=spec_tuples,
                    wnd_sz=wnd_sz,
                    dt=dt,
                    feature_extraction=feature_extraction,
                    online=online)

                # Keep track of how many on- and off-type windows we extracted so far
                curr_on += len(on_windows_tmp)
                curr_off += len(off_windows_tmp)

                on_windows.extend(on_windows_tmp)
                off_windows.extend(off_windows_tmp)

                if curr_on >= on_num and curr_off >= off_num:
                    break

            # Make sure that we don't return too many windows
            if curr_on >= on_num and curr_off >= off_num:
                on_windows = random.sample(on_windows, curr_on)
                off_windows = random.sample(off_windows, curr_off)
            # If one type has less tuples than desired, compensate by adding more windows
            # from the other type. Like this we ensure that we return exactly 'limit' many windows
            elif curr_on >= on_num:
                on_windows = random.sample(on_windows, min(len(on_windows), limit - len(off_windows)))
            elif curr_off >= off_num:
                off_windows = random.sample(off_windows, min(len(off_windows), limit - len(on_windows)))

            windows = on_windows + off_windows
            labels = [1] * len(on_windows) + [0] * len(off_windows)

            # Shuffle the windows, and labels arrays
            combined = list(zip(windows, labels))
            combined = random.sample(combined, len(combined))
            windows = list(map(lambda tup: tup[0], combined))
            labels = list(map(lambda tup: tup[1], combined))

            results[wnd_sz][bird_name]["X"] = windows
            results[wnd_sz][bird_name]["y"] = labels

            real_configs[wnd_sz][bird_name]["total"] = len(combined)
            real_configs[wnd_sz][bird_name]["on_frac"] = len(on_windows) / len(combined)

    return results, real_configs


def store_birds(name, bird_data):
    dump(BIRD_DATA_PATH + "bird_data_"+name, bird_data)
    np.save(BIRD_DATA_PATH + "bird_data_" + name, bird_data)


def load_birds(name):
    file_name = BIRD_DATA_PATH + "bird_data_" + name + ".npy"
    if path.isfile(file_name):
        return np.load(file_name,  allow_pickle=True)
    return None


def store_dataset(name, results, configs=None):
    """
    This function is written to store dataset dictionaries like the ones returned by the function
    'create_windows'. However, it doesn't check whether its input follows these rules, so you can also
    use it to store simple 'X', 'y' pairs.
    """
    pair = [results, configs]
    np.save(DATASET_PATH + "dataset_" + name, pair)


def load_dataset(name):
    file_name = DATASET_PATH + "dataset_" + name + ".npy"
    if path.isfile(file_name):
        (results, configs) = np.load(file_name,  allow_pickle=True)
        if configs is None:
            return results
        return results, configs
    return None


def flatten_windows_dic(windows_dic, bird_names=None):
    """
    Takes a 'window_dic' like for example the one returned by 'create_windows' (without the first level
    which specifies the window size) and combine the datasets of all birds specified by 'bird_names'
    to two single arrays X and y. These two arrays can then be used for training
    """
    if bird_names is None:
        bird_names = windows_dic.keys()

    X = []
    y = []

    for bird_name in bird_names:
        bird_windows = windows_dic[bird_name]
        X.extend(bird_windows['X'])
        y.extend(bird_windows['y'])

    return X, y


class BirdDataLoader:

    def __init__(self, train, validation, test, network_type="cnn", normalize_input=True, online=False):
        # network type can be either 'cnn' or 'rnn'

        self.X_train, self.X_val, self.X_test = torch.tensor(train[0]), torch.tensor(validation[0]), torch.tensor(test[0])
        self.y_train, self.y_val, self.y_test = torch.tensor(train[1]), torch.tensor(validation[1]), torch.tensor(test[1])

        self.input_sz = self.X_train.shape[2]
        if online:
            self.wnd_sz = self.input_sz
        else:
            self.wnd_sz = int(self.input_sz / 2)
        self.nfreq = self.X_train.shape[1]

        if normalize_input:
            self.mean = torch.mean(self.X_train.float())
            self.std = torch.std(self.X_train.float())
            self.X_train = normalize(self.X_train.float())
            self.X_val = normalize(self.X_val.float())
            self.X_test = normalize(self.X_test.float())
        else:
            self.mean = 0.0
            self.std = 1.0

        if network_type == "cnn":
            dimensions = (-1, 1, 128, self.input_sz)
        elif network_type == "rnn":
            dimensions = (-1, 128, self.input_sz)

        self.train_dataset = TensorDataset(torch.reshape(self.X_train, dimensions), self.y_train)
        self.val_dataset = TensorDataset(torch.reshape(self.X_val, dimensions), self.y_val)
        self.test_dataset = TensorDataset(torch.reshape(self.X_test, dimensions), self.y_test)

    def get_data_loader(self, dset, shuffle, num_workers, batch_size):
        if dset == "train":
            dataloader = DataLoader(dataset=self.train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        elif dset == "val":
            dataloader = DataLoader(dataset=self.val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        elif dset == "test":
            dataloader = DataLoader(dataset=self.test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
        else:
            assert False, "Unknown dset"
        return dataloader
