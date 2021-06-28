from utils import (
    # Constants
    DEVICE, BIRD_NAMES, DATA_PATH, MODEL_PATH, PREDICTIONS_PATH,

    # Data handling functions
    download, load_bird_data, extract_labelled_spectrograms, train_test_split,
    extract_birds, create_windows, store_birds, load_birds, store_dataset,
    load_dataset, flatten_windows_dic, standardize_data
)

from classifiers import (
    # For the cnn
    train_CNN, predict_syllables_CNN, load_cnn, wrap_cnn,

    # For the rnn
    train_RNN, predict_syllables_RNN, load_rnn, wrap_rnn
)

# First choose one of the different CNNs in the 'MODEL_PATH' folder
cnn_name = "task1_cnn_g17y2_wnd_26_online_False_size_81876"

# Load the specified CNN
cnn = load_cnn(MODEL_PATH + cnn_name, wnd_sz=26, online=False)

# Wrap the CNN. Should it be able to predict whole spectrograms? => "mode='for_spectrograms'"
# Should it only predict a list of extracted windows? => "mode='for_windows'".
cnn = wrap_cnn(cnn, mode="for_spectrograms")

# Now it's time to load the data. There are two options how this can be done:
#
# 1. Load the data from wherever and however you like. In this case, just make sure that in the end
#    you have a numpy array containing all spectrograms. I.e. the shape should look like this:
#
#    [
#       [ [1,2,3,4,5],
#         [6,7,8,9,0],
#         ...         ],
#
#       [ [1,2,3,4,5],
#         [6,7,8,9,0]
#         ...         ],
#
#       ...
#    ]
#
# 2. Use the provided methods. If you plan to use the data which you already gave us for training,
#    there's nothing to do. If you want to add data of a new bird / data which we haven't received
#    for training, you should put a 'my_bird_train.mat' file into the
#    'Data/train_dataset/spectrograms' folder and another 'annotations_my_bird_train.mat' file into
#    'Data/train_dataset/annotations' folder. Here, "my_bird" is a custom name which you can choose.
#    After this step you should add your custom bird name to the global list storing the names of the
#    bird. This variable is called 'BIRD_NAMES' and can be found in the file 'utils.py'

# If you chose option 1, load your data here:

# If you chose option 2, this is an example of how to load the data of one or several birds using
# our provided methods:
bird_data = load_bird_data(names=["g17y2", "g4p5"])

g17y2_spectrograms = bird_data["g17y2"]["Xs_train"]
g4p5_spectrograms = bird_data["g4p5"]["Xs_train"]

# If the bird data you chose also contains annotations, you can access them like this.
# If the provided data doesn't contain annotations, comment out the following two lines:
g17y2_tuples = bird_data["g17y2"]["tuples"]
g4p5_tuples = bird_data["g4p5"]["tuples"]

# Finally, make predictions for the data
# Iterate over all spectrograms and make predictions for each spectogram
predictions_g17y2 = []
predictions_g4p5 = []

print("Making predictions for the first 40 spectrograms of bird g17y2")
for index, spectrogram in enumerate(g17y2_spectrograms[:40]):
    if index % 20 == 0:
        print(f"{index}/{len(g17y2_spectrograms)}")
    predictions_g17y2.extend(
        cnn(g17y2_spectrograms[index], index))

print("Making predictions for the first 40 spectrograms of bird g4p5")
for index, spectrogram in enumerate(g4p5_spectrograms[:40]):
    if index % 20 == 0:
        print(f"{index}/{len(g4p5_spectrograms)}")
    predictions_g4p5.extend(
        cnn(g4p5_spectrograms[index], index))

# The predictions are lists of tuples. Each tuple has the form (start, end, index)
# meaning that the 'index'th spectrogram contains a syllable spanning from columns
# [start, end]
print("Showing all predictions for spectrogram 0 of bird g17y2:")
print([tup for tup in predictions_g17y2 if tup[2] == 0])
