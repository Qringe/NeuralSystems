import numpy as np
import os.path as path
from scipy.io import loadmat
from joblib import dump, load

from utils import (
    # Constants
    device, bird_names, data_path, model_path, predictions_path,

    # Data handling functions
    download, load_bird_data, extract_labelled_spectograms, train_test_split,
    extract_birds, create_windows, store_birds, load_birds, store_dataset,
    load_dataset, flatten_windows_dic,

    # Other stuff
    hash_spectograms, score_predictions
)

from classifiers import (
    # For the svm
    train_SVM, predict_syllables_SVM,
    
    # For the cnn
    train_CNN, predict_syllables_CNN, load_cnn, wrap_cnn,

    # For the rnn
    train_RNN, predict_syllables_RNN, load_rnn, wrap_rnn
)

def predict_dataset(model, spectograms, ground_truth_tuples, name, read_cache=True, write_cache=True):
    """
    Makes predictions for the specified dataset consisting of spectograms (parameter 'spectograms')
    and labelled tuples (parameter 'ground_truth_tuples') of the form (on, off, SETindex, bird_name)

    If 'read_cache==True', then the
    function first checks, whether the results have already been computed (i.e.
    whether a file with name 'name' already exists) and if so, loads the data.

    If 'write_cache==True' then the function will store the computed results.
    """
    base = path.dirname(path.abspath(__file__))
    p = path.join(base,predictions_path + f"predictions_{name}.data")

    if read_cache and path.isfile(p):
        return load(p)

    results = []
    # Iterate over all spectograms
    for index, spec in enumerate(spectograms):
        if index % 20 == 0:
            print(index,"/",len(spectograms))

        # Generate predictions for current spectogram. Make sure that they are sorted
        y_pred = sorted(model(spec,index),
                        key=lambda t: (t[2],t[0],t[1]))

        # Fetch the true tuples for this spectogram. Make sure that they are sorted
        y_true = sorted([tup for tup in ground_truth_tuples if tup[2] == index],
                        key=lambda t: (t[2],t[0],t[1]))

        results.append({"y_true": y_true,"y_pred": y_pred})

    # Store the results, if write_cache = True
    if write_cache:
        dump(results,p)

    return results

def predict_whole_dataset(model, name, read_cache=True, write_cache=True):
    """
    Loads the whole dataset and makes predictions. 
    
    If 'read_cache==True', then the
    function first checks, whether the results have already been computed (i.e.
    whether a file with name 'name' already exists) and if so, loads the data.

    If 'write_cache==True' then the function will store the computed results.
    """
    base = path.dirname(path.abspath(__file__))
    p = path.join(base,data_path + f"predictions_{name}.data")

    if read_cache and path.isfile(p):
        return load(p)

    # Download the data if necessary
    download()
    
    # Load the whole data
    Xs_train = loadmat(path.join(base, data_path + "train_dataset/spectrograms/g17y2_train.mat"))["Xs_train"][0]
    annotations = loadmat(path.join(base, data_path + "train_dataset/annotations/annotations_g17y2_train.mat"))["annotations_train"][0]

    SETindex = annotations["y"][0]["SETindex"][0][0][0]-1
    ons = annotations["y"][0]["onset_columns"][0][0][0]-1
    offs = annotations["y"][0]["offset_columns"][0][0][0]-1
    ground_truth_tuples = [(a,b,c) for a,b,c in zip(ons,offs,SETindex)]

    return predict_dataset(model, Xs_train, ground_truth_tuples, name)    

def analyze_errors(predictions):
    """
    This function compares the vocal/non-vocal predictions of a model to the true
    vocal/non-vocal intervals and keeps track of 4 different types of errors:
        1. Border error: The prediction is a bit longer/shorter than the real error.
            - Severity: Small
        2. Noise error: In this case the model misclassified noise as a syllable
            - Severity: Medium
        3. Skip error: In this case the model misclassified a real syllable as noise
           (i.e. the model skipped a real syllable)
            - Severity: Medium
        4. Split error: Here the model did accidentally split a syllable into 2 syllables
            - Severity: Large
    """
    ## A few helper functions which each operate on two lists of tuples
    def set_intersect(a, b):
        """
        Takes two lists of tuples as input and returns all tuples from 'a' whose ranges intersect
        with at least one tuple from 'b' 
        """
        return [t_a for t_a in a if 
            any(map(lambda t_b: 
                (t_a[0] <= t_b[0] and t_a[1] >= t_b[0]) or \
                (t_a[0] <= t_b[1] and t_a[1] >= t_b[1]) or \
                (t_a[0] >= t_b[0] and t_a[1] <= t_b[1]) 
                ,b))]

    def set_minus(a,b):
        """
        Takes two lists of tuples as input and returns all tuples from 'a' which don't
        intersect with any tuple from 'b'
        """
        intersect_with_b = set_intersect(a,b)
        return [t_a for t_a in a if t_a not in intersect_with_b]

    def set_contains(a,b):
        """
        Takes two lists of tuples as input and returns all tuples from 'a' whose interval
        is contained in at least one tuple from 'b'
        """
        return [t_a for t_a in a if 
            any(map(lambda t_b: t_a[0] >= t_b[0] and t_a[1] <= t_b[1],b))]

    # Store all tuples which are erroneous in their respective arrays
    type1_tuples = []
    type2_tuples = []
    type3_tuples = []
    type4_tuples = []

    # Store the amount of times each error occurs in these variables
    type1_amount = type2_amount = type3_amount = type4_amount = 0

    accuracies = []

    # Iterate through all spectograms
    for index, result in enumerate(predictions):
        y_true = result['y_true']
        y_pred = result['y_pred']

        # If we don't have any lables, there is nothing to analyze
        if y_true == []:
            continue

        ## Find all errors of type 1 (Border errors)
        overlap = set_intersect(y_pred, y_true)
        type1 = [t for t in y_pred if t in overlap and t not in y_true]
        type1_amount += len(type1)
        type1_tuples.extend(type1)

        ## Find all errors of type 2 (Noise errors)
        type2 = set_minus(y_pred, y_true)
        type2_amount += len(type2)
        type2_tuples.extend(type2)

        ## Find all errors of type 3 (Skip errors)
        type3 = set_minus(y_true, y_pred)
        type3_amount += len(type3)
        type3_tuples.extend(type3)

        ## Find all errors of type 4 (Split errors)
        # For all consecutive pairs of tuples 'a','b' in 'y_pred', get the interval 
        # of the space between 'a' and 'b'
        consecutive_tuples = list(zip(y_pred, y_pred[1:]))
        consecutive_tuples = [(t[0][1],t[1][0],t[0][2]) for t in consecutive_tuples]
        type4 = set_contains(consecutive_tuples,y_true)
        type4_amount += len(type4)
        type4_tuples.extend(type4)

        accuracies.append(score_predictions(y_true, y_pred))

    return {'type1_amount': type1_amount,
            'type2_amount': type2_amount,
            'type3_amount': type3_amount,
            'type4_amount': type4_amount,
            'type1_tuples': type1_tuples,
            'type2_tuples': type2_tuples,
            'type3_tuples': type3_tuples,
            'type4_tuples': type4_tuples,
            'accuracy_mean' : np.mean(accuracies),
            'accuracy_std' : np.std(accuracies)}

def compare_classifiers(dataset = None, model_dic = None, print_summary = True):
    """
    Take the different classifiers and generates predictions for the specified dataset.
    The predictions are then analyzed and the different types of errors get listed.

    'model_dic' is a dictionary containing a mapping of model names to models:
        {
            "model_name1" : model1,
            "model_name2" : model2,
            ...
        }


    Note: This function assumes that the classifiers have already been trained before.
    """
    if model_dic == None:
        raise Exception("You need to specify at least one model!")

    # If dataset is not specified, use total dataset over all birds
    if dataset == None:
        dataset = load_bird_data()

    bird_names = [key for key in list(dataset.keys()) if type(key) == np.str_ or type(key) == str]

    summary_total = {}
    for model_name in model_dic.keys():
        summary_total[model_name] = {}

    # Iterate through all specified birds and generate predictions
    for bird_name in bird_names:
        bird_data = dataset[bird_name]
        spectograms = bird_data['Xs_train']
        tuples = bird_data['tuples']

        results = {}
        errors = {}

        # Make predictions for this specific bird
        for model_name in model_dic.keys():
            print(f"Make predictions for bird {bird_name} using the model {model_name}")
            
            # Use a hash value in the name to check if the input file already exists or not
            name = f"{model_name}_{bird_name}_hash_{hash_spectograms(spectograms)}"
            results[model_name] = predict_dataset(model=model_dic[model_name], 
                spectograms=spectograms, ground_truth_tuples=tuples, 
                name=name)

            # Analyze the errors of the different predictors
            errors[model_name] = analyze_errors(results[model_name])

            # Add the errors of this specific model on this specific bird to the summary
            for error_type in range(1,5,1):
                error_name = f"type{error_type}_amount"
                summary_total[model_name][error_name] = summary_total[model_name].get(error_name,0) + \
                                                                errors[model_name][error_name]
            # Add the accuracies per model
            summary_total[model_name]['accuracy_mean'] = summary_total[model_name].get('accuracy_mean',[]) \
                        + [errors[model_name]['accuracy_mean']]
            summary_total[model_name]['accuracy_std'] = summary_total[model_name].get('accuracy_std',[]) \
                        + [errors[model_name]['accuracy_std']]

        summary_total[bird_name] = errors

    if not print_summary:
        return summary_total
    
    # Print a summary of the analysis results
    line = "=" * 50
    def header1(text): print(line); print(text); print(line)
    def header2(text): print(text+ ":"); print("-" * (len(text) + 1))
    def print_overview(error_dic):
        print("Type 1 errors (Border errors): ", error_dic["type1_amount"])
        print("Type 2 errors (Noise errors) : ", error_dic["type2_amount"])
        print("Type 3 errors (Skip errors)  : ", error_dic["type3_amount"])
        print("Type 4 errors (Split errors) : ", error_dic["type4_amount"],"\n")
        print("Mean accuracy: " + str(np.mean(error_dic['accuracy_mean'])))
        print("Mean std of accuracy: " + str(np.mean(error_dic['accuracy_std'])))
        print()

    print()
    header1("SUMMARY")
    header1("TOTAL")
    for model_name in model_dic.keys():
        header2(model_name)
        print_overview(summary_total[model_name])
    
    for bird_name in bird_names:
        header1("BIRD "+bird_name)
        for model_name in model_dic.keys():
            header2(model_name)
            print_overview(summary_total[bird_name][model_name])

    return summary_total

if __name__ == "__main__":

    # Set some general parameters
    use_feature_extraction = False
    wnd_sz = 20
    limit = 70000

    # Some RNN parameters
    network_type = "gru"    # Choose from {'rnn', 'lstm', 'gru'}
    num_layers = 1
    hidden_size = 100

    # The paths to the models
    base = path.dirname(path.abspath(__file__))
    cnn_name = "CNN_features_%s_wnd_sz_%s_limit_%s_v02.model" % (use_feature_extraction,wnd_sz,limit)
    cnn_path = path.join(base, model_path+cnn_name)
    rnn_name = "RNN_type_%s_num_layers_%s_hidden_size_%s_features_%s_wnd_sz_%s_limit_%s.model" % (network_type, num_layers, hidden_size, use_feature_extraction,wnd_sz,limit)
    rnn_path = path.join(base, model_path+rnn_name)

    if not (path.isfile(cnn_path) and path.isfile(rnn_path)):
        # Load the data and get all labelled spectograms
        bird_data = load_bird_data(names = ["g17y2"])
        data_labelled, _ = extract_labelled_spectograms(bird_data)

        # Perform a train-validation-test split
        data_train, data_test = train_test_split(bird_data = data_labelled, configs = 0.33, seed = 42)
        data_val, data_test = train_test_split(bird_data = data_test, configs = 0.5, seed = 42)

        # Extract the windows from the spectograms
        windows_train, _ = create_windows(bird_data = data_train, wnd_sizes = wnd_sz, limits = limit, on_fracs = 0.7, dt = 5, seed = 42)
        windows_val, _ = create_windows(bird_data = data_val, wnd_sizes = wnd_sz, limits = limit/2, on_fracs = 0.7, dt = 5, seed = 42)
        windows_test, _ = create_windows(bird_data = data_test, wnd_sizes = wnd_sz, limits = limit/2, on_fracs = 0.7, dt = 5, seed = 42)

        X_train, y_train = flatten_windows_dic(windows_train[wnd_sz])
        X_val, y_val = flatten_windows_dic(windows_val[wnd_sz])
        X_test, y_test = flatten_windows_dic(windows_test[wnd_sz])

        dataset = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val), 
            "test": (X_test, y_test)
        }

        if not path.isfile(cnn_path):
            cnn = train_CNN(dataset,cnn_name)
        if not path.isfile(rnn_path):
            rnn = train_RNN(dataset,rnn_name, network_type = network_type, hidden_size = hidden_size, num_layers=num_layers)

    # Load the CNN
    cnn = load_cnn(cnn_path, wnd_sz)
    cnn = wrap_cnn(cnn, mode="for_spectograms")

    # Load the RNN
    rnn = load_rnn(rnn_path, network_type, hidden_size=hidden_size, num_layers = num_layers, device=device)
    rnn = wrap_rnn(rnn, mode="for_spectograms")

    model_dic = {
        "cnn_v01" : cnn,
        "rnn_v01" : rnn
    }

    compare_classifiers(dataset = None, model_dic = model_dic, print_summary = True)