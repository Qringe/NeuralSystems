import numpy as np
import os.path as path

from utils import (
    # Constants
    device, bird_names, data_path, model_path, predictions_path,

    # Data handling functions
    download, load_bird_data, extract_labelled_spectograms, train_test_split,
    extract_birds, create_windows, store_birds, load_birds, store_dataset,
    load_dataset, flatten_windows_dic
)

from classifiers import (
    # For the svm
    train_SVM, predict_syllables_SVM,
    
    # For the cnn
    train_CNN, predict_syllables_CNN, load_cnn, wrap_cnn,

    # For the rnn
    train_RNN, predict_syllables_RNN, load_rnn, wrap_rnn
)

from analyze_errors import compare_classifiers

## NOTE: These are the existing bird names ["g17y2", "g19o10", "g4p5", "R3428"] as stord in 
# the 'bird_names' variable

## Step 0: First make sure that you have all the necessary files:
download()

## STEP 1: Load the data of one or several birds. Let's take two for our example
bird_data = load_bird_data(names = ["g17y2", "g19o10"])

# The resulting dictionary 'bird_data' has the format:
#   {
#       "g17y2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples, "ids_labelled" : ids_labelled},
#       "g19o10": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples, "ids_labelled" : ids_labelled},
#   }

## STEP 2: Not all spectograms are labelled. For training we can only use labelled spectograms.
# Therefore, extract all labelled spectograms
data_labelled, data_unlabelled = extract_labelled_spectograms(bird_data)

# The resulting two dictionaries have the same form like 'bird_data', just that one contains all the labelled
# spectograms and the other all the unlabelled spectograms. Furthermore, the field "ids_labelled" is not present
# anymore as it was only used to extract the labelled spectograms

## STEP 3: Create a train-test split
# Let's use 20% of all spectograms of bird "g17y2" and 80% of all spectograms of bird "g19o10" for testing
configs = {
    "g17y2" : 0.2,
    "g19o10" : 0.8
}
data_train, data_test = train_test_split(bird_data = data_labelled, configs = configs, seed = 42)

# The resulting two dictionaries have again the following format:
#   {
#       "g17y2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
#       "g19o10": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
#   }

## STEP 4: Extract windows from the spectograms for different window sizes
#  Let's define two different window sizes for the moment
wnd_sizes = [20,40]

# Let's use 40'000 windows of bird "g17y2" and about 50% of all possible windows of bird "g19o10"
limits = {
    "g17y2" : 40000,
    "g19o10" : 0.5
}

# Furthermore, for bird "g17y2" let's extract about 50% on-windows and 50% off-windows, while for bird
# "g19o10" we would like to see 12 on-windows. This can be specified as follows:
on_fracs = {
    "g17y2" : 0.5,
    "g19o10" : 12
}
windows_train, real_sizes_train = create_windows(
        bird_data = data_train, 
        wnd_sizes = wnd_sizes, 
        limits = limits,  
        on_fracs = on_fracs, 
        dt = 5,
        seed = 42)

# The resulting dictionary 'windows_train' has the following format:
#   {
#       20: {
#               "g17y2": {"X" : windows, "y" : labels},
#               "g19o10": {"X" : windows, "y" : labels},
#           },
#       40: {
#               "g17y2": {"X" : windows, "y" : labels},
#               "g19o10": {"X" : windows, "y" : labels},
#           }
#   }

# The dictionary 'real_sizes' shows you how well the function was able to implement the restrictions given by
# 'limits' and 'on_fracs'. It looks like this:
#   {
#    20: {
#        'g17y2': {'total': 40298, 'on_frac': 0.5}, 
#        'g19o10': {'total': 1956, 'on_frac': 0.24}
#        }, 
#    40: {
#        'g17y2': {'total': 40293, 'on_frac': 0.5}, 
#        'g19o10': {'total': 1928, 'on_frac': 0.26}
#        }
#   }
# As you can see, for 'g17y2', the window limit of 40000 and the 'on_frac' value of 0.5 were quite well implemented.
# For the bird 'g19o10' we see that it has much more than only 12 on-windows (about 19300 actually). The reason for this
# is that there weren't enough off-windows such that #on-windows + #off-windows = limit, therefore, the function took all 
# existing off-windows and filled the rest up with on-windows.

# Let's do the same thing for the test_data.
windows_test, real_sizes_test = create_windows(
        bird_data = data_test, 
        wnd_sizes = wnd_sizes, 
        limits = 1.0,  
        on_fracs = 0.5, 
        dt = 5,
        seed = 42)

## STEP 5: Let's store the dataset such that we don't have to recompute it everytime
store_dataset(name="two_birds", results=windows_train, configs = real_sizes_train)

# Let's reload it now
windows_train, real_sizes_train = load_dataset(name="two_birds")

## STEP 6: Let's train two models on this dataset
# First, let's restrict us to the data extracted with window_size 20
windows_train = windows_train[20]
windows_test = windows_test[20]

# Now, let's transform the dictionary into an array, which can be used to train our models
X_train, y_train = flatten_windows_dic(windows_train)
X_test, y_test = flatten_windows_dic(windows_test)

# We can now put the whole dataset together:
dataset = {
    "train": (X_train, y_train),
    "validation": (X_train, y_train), # I know we shouldn't use the training dataset as validation set, but it's only now that I realized that I need this. Furthermore, it's late and I'm tired. So, forgive me. :)
    "test": (X_test, y_test)
}

# Let's define a few parameters for our model
# Set some general parameters
use_feature_extraction = False
wnd_sz = 20
limit = 70000

# Some RNN parameters
network_type = "gru"    # Choose from {'rnn', 'lstm', 'gru'}
num_layers = 1
hidden_size = 100

# Train a CNN and an RNN
cnn = train_CNN(dataset,model_name = "test_cnn")
rnn = train_RNN(dataset,model_name = "test_rnn", network_type = network_type, hidden_size = hidden_size, num_layers=num_layers)


# Load the CNN
cnn = load_cnn(model_path + "test_cnn", wnd_sz)
cnn = wrap_cnn(cnn, mode="for_spectograms")

# Load the RNN
rnn = load_rnn(model_path + "test_rnn", network_type, hidden_size=hidden_size, num_layers = num_layers, device=device)
rnn = wrap_rnn(rnn, mode="for_spectograms")

model_dic = {
    "cnn_v01" : cnn,
    "rnn_v01" : rnn
}

# Compare the two classifiers
compare_classifiers(dataset = data_test, model_dic = model_dic, print_summary = True)
