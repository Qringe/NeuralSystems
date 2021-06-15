import numpy as np
import os.path as path
import gc
from joblib import dump, load
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils import (
    # Constants
    DEVICE, BIRD_NAMES, DATA_PATH, MODEL_PATH, PREDICTIONS_PATH, TOLERANCE,

    # Data handling functions
    load_bird_data, extract_labelled_spectograms, train_test_split,
    extract_birds, create_windows, store_birds, load_birds, store_dataset,
    load_dataset, flatten_windows_dic, standardize_data,

    # Other stuff
    hash_spectograms, tuples2vector, score_predictions
)

from classifiers import (
    # For the cnn
    train_CNN, predict_syllables_CNN, load_cnn, wrap_cnn,

    # For the rnn
    train_RNN, predict_syllables_RNN, load_rnn, wrap_rnn,

    # For transfer learning
    get_transfer_learning_models_CNN, get_transfer_learning_models_RNN
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
    p = path.join(base, PREDICTIONS_PATH + f"predictions_{name}.data")

    if read_cache and path.isfile(p):
        return load(p)

    results = []
    # Iterate over all spectograms
    for index, spec in enumerate(spectograms):
        if index % 20 == 0:
            gc.collect()
            print(index, "/", len(spectograms))

        # Generate predictions for current spectogram. Make sure that they are sorted
        y_pred = sorted(model(spec, index),
                        key=lambda t: (t[2], t[0], t[1]))

        # Fetch the true tuples for this spectogram. Make sure that they are sorted
        y_true = sorted([tup for tup in ground_truth_tuples if tup[2] == index],
                        key=lambda t: (t[2], t[0], t[1]))

        results.append({"y_true": y_true, "y_pred": y_pred, "spectogram_length": len(spec[0])})

    # Store the results, if write_cache = True
    if write_cache:
        dump(results, p)

    return results


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

    Furthermore, it displays a few other metrics such as a score (used for grading),
    how many columns were predicted correctly in total and other metrics
    """
    # A few helper functions which each operate on two lists of tuples
    def set_intersect(a, b):
        """
        Takes two lists of tuples as input and returns all tuples from 'a' whose ranges intersect
        with at least one tuple from 'b'
        """
        return [t_a for t_a in a if
                any(map(lambda t_b:
                        (t_a[0] <= t_b[0] and t_a[1] >= t_b[0]) or
                    (t_a[0] <= t_b[1] and t_a[1] >= t_b[1]) or
                    (t_a[0] >= t_b[0] and t_a[1] <= t_b[1]), b))]

    def set_minus(a, b):
        """
        Takes two lists of tuples as input and returns all tuples from 'a' which don't
        intersect with any tuple from 'b'
        """
        intersect_with_b = set_intersect(a, b)
        return [t_a for t_a in a if t_a not in intersect_with_b]

    def set_contains(a, b):
        """
        Takes two lists of tuples as input and returns all tuples from 'a' whose interval
        is contained in at least one tuple from 'b'
        """
        return [t_a for t_a in a if
                any(map(lambda t_b: t_a[0] >= t_b[0] and t_a[1] <= t_b[1], b))]

    # Store all tuples which are erroneous in their respective arrays
    type1_tuples = []
    type2_tuples = []
    type3_tuples = []
    type4_tuples = []

    # Store the amount of times each error occurs in these variables
    type1_amount = type2_amount = type3_amount = type4_amount = 0

    accuracies = []
    prediction_tuples = []
    prediction_vector = []
    solution_tuples = []
    solution_vector = []

    # Iterate through all spectograms
    for index, result in enumerate(predictions):
        y_true = result['y_true']
        y_pred = result['y_pred']
        length = result['spectogram_length']

        prediction_tuples.append((y_pred, length))
        solution_tuples.append((y_true, length))

        # If we don't have any lables, there is nothing to analyze
        if y_true == []:
            type1_tuples.append([])
            type2_tuples.append(y_pred)
            type3_tuples.append([])
            type4_tuples.append([])
            continue

        # Find all errors of type 1 (Border errors)
        # overlap = set_intersect(y_pred, y_true)
        # type1 = [t for t in y_pred if t in overlap and t not in y_true]
        type1 = [t for t in y_pred if len(
            {t2 for t2 in y_true if abs(t[0] - t2[0]) <= TOLERANCE and abs(t[1] - t2[1]) <= TOLERANCE}
            ) == 0]
        type1_amount += len(type1)
        type1_tuples.append(type1)

        # Find all errors of type 2 (Noise errors)
        type2 = set_minus(y_pred, y_true)
        type2_amount += len(type2)
        type2_tuples.append(type2)

        # Find all errors of type 3 (Skip errors)
        type3 = set_minus(y_true, y_pred)
        type3_amount += len(type3)
        type3_tuples.append(type3)

        # Find all errors of type 4 (Split errors)
        # For all consecutive pairs of tuples 'a','b' in 'y_pred', get the interval
        # of the space between 'a' and 'b'
        consecutive_tuples = list(zip(y_pred, y_pred[1:]))
        consecutive_tuples = [(t[0][1], t[1][0], t[0][2]) for t in consecutive_tuples]
        type4 = set_contains(consecutive_tuples, y_true)
        type4_amount += len(type4)
        type4_tuples.append(type4)

        accuracies.append(score_predictions(y_true, y_pred, tolerance=TOLERANCE))

    # Compute a few standard error metrics
    # First transform the tuples into a single array of 0s and 1s representing local/
    # non-vocal columns
    prediction_vector = tuples2vector(prediction_tuples)
    solution_vector = tuples2vector(solution_tuples)

    # Compute error metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        solution_vector,
        prediction_vector,
        average="binary"
    )
    accuracy = accuracy_score(solution_vector, prediction_vector)

    return {'type1_amount': type1_amount,
            'type2_amount': type2_amount,
            'type3_amount': type3_amount,
            'type4_amount': type4_amount,
            'type1_tuples': type1_tuples,
            'type2_tuples': type2_tuples,
            'type3_tuples': type3_tuples,
            'type4_tuples': type4_tuples,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'score_mean': np.mean(accuracies),
            'score_std': np.std(accuracies)}


def compare_classifiers(dataset=None, model_dic=None, print_summary=True):
    """
    Take the different classifiers and generates predictions for the specified dataset.
    The predictions are then analyzed and the different types of errors get listed.

    IF 'dataset' is 'None', the data of all birds is loaded. Else it should have the form:
        {
            "bird_name1": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
            "bird_name2": {"Xs_train" : Xs_train, "tuples" : ground_truth_tuples},
            ...
        }
    where "Xs_train" is a list of spectograms and "tuples" a list of tuples of the form (on,off,SETindex,bird_name) denoting
    the start (on) and end (off) of a syllable, and the spectogram index (SETindex) of said syllable, and the name of the bird
    (bird_name) from which the syllable stems from.

    'model_dic' is a dictionary containing a mapping of model names to models:
        {
            "model_name1" : model1,
            "model_name2" : model2,
            ...
        }

    Note: This function assumes that the classifiers have already been trained before.
    """
    if model_dic is None:
        raise Exception("You need to specify at least one model!")

    # Implemented error metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'score_mean', 'score_std']

    # If dataset is not specified, use total dataset over all birds
    if dataset is None:
        dataset = load_bird_data()
        dataset, _ = extract_labelled_spectograms(dataset)

    bird_names = [key for key in list(dataset.keys()) if type(key) == np.str_ or type(key) == str]

    summary_total = {}
    for model_name in model_dic.keys():
        summary_total[model_name] = {}

    # Iterate through all specified birds and generate predictions
    for bird_name in bird_names:
        gc.collect()
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
            for error_type in range(1, 5, 1):
                error_name = f"type{error_type}_amount"
                summary_total[model_name][error_name] = summary_total[model_name].get(error_name, 0) + \
                    errors[model_name][error_name]
            # Add the other score metrics per model
            for metric in metrics:
                summary_total[model_name][metric] = summary_total[model_name].get(metric, []) \
                    + [errors[model_name][metric]]

        summary_total[bird_name] = errors

    if not print_summary:
        return summary_total

    # Print a summary of the analysis results
    line = "=" * 50
    def header1(text): print(line); print(text); print(line)
    def header2(text): print(text + ":"); print("-" * (len(text) + 1))

    def print_overview(error_dic):
        print("Type 1 errors (Border errors): ", error_dic["type1_amount"])
        print("Type 2 errors (Noise errors) : ", error_dic["type2_amount"])
        print("Type 3 errors (Skip errors)  : ", error_dic["type3_amount"])
        print("Type 4 errors (Split errors) : ", error_dic["type4_amount"], "\n")
        for metric in metrics:
            print(metric + ": " + str(np.mean(error_dic[metric])))
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
    limit = 80000
    standardize = False
    online = False

    # If you only want to train the cnn stuff, set this to False
    rnn_should_be_trained = False

    # Some RNN parameters
    network_type = "gru"    # Choose from {'rnn', 'lstm', 'gru'}
    num_layers = 1
    hidden_size = 100

    # The paths to the models
    base = path.dirname(path.abspath(__file__))
    cnn_name = "CNN_features_%s_wnd_sz_%s_limit_%s_v02.model" % (use_feature_extraction, wnd_sz, limit)
    cnn_path = path.join(base, MODEL_PATH+cnn_name)
    rnn_name = "RNN_type_%s_num_layers_%s_hidden_size_%s_features_%s_wnd_sz_%s_limit_%s.model" % (
        network_type, num_layers, hidden_size, use_feature_extraction, wnd_sz, limit)
    rnn_path = path.join(base, MODEL_PATH+rnn_name)

    if not (path.isfile(cnn_path) and (not rnn_should_be_trained or path.isfile(rnn_path))):
        # Load the data and get all labelled spectograms
        bird_data = load_bird_data(names=["g17y2"])

        if standardize:
            bird_data = standardize_data(bird_data, coarse_mode="per_spectogram", fine_mode="scalar")
        data_labelled, _ = extract_labelled_spectograms(bird_data)

        # Perform a train-validation-test split
        data_train, data_test = train_test_split(bird_data=data_labelled, configs=0.33, seed=42)
        data_val, data_test = train_test_split(bird_data=data_test, configs=0.5, seed=42)

        # Extract the windows from the spectograms
        windows_train, _ = create_windows(bird_data=data_train, wnd_sizes=wnd_sz, limits=limit, on_fracs=0.5, dt=5, seed=42)
        windows_val, _ = create_windows(bird_data=data_val, wnd_sizes=wnd_sz, limits=int(limit/2), on_fracs=0.5, dt=5, seed=42)
        windows_test, _ = create_windows(bird_data=data_test, wnd_sizes=wnd_sz, limits=int(limit/2), on_fracs=0.5, dt=5, seed=42)

        X_train, y_train = flatten_windows_dic(windows_train[wnd_sz])
        X_val, y_val = flatten_windows_dic(windows_val[wnd_sz])
        X_test, y_test = flatten_windows_dic(windows_test[wnd_sz])

        dataset = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test)
        }

        if not path.isfile(cnn_path):
            cnn = train_CNN(dataset, cnn_name, normalize_input=True, online=online)
        if rnn_should_be_trained and not path.isfile(rnn_path):
            rnn = train_RNN(dataset, rnn_name, network_type=network_type, hidden_size=hidden_size,
                            num_layers=num_layers, normalize_input=True, online=online)

    # Load the CNN
    cnn = load_cnn(cnn_path, wnd_sz, online=online)
    if rnn_should_be_trained:
        rnn = load_rnn(rnn_path, network_type, nfreq=128, hidden_size=hidden_size, num_layers=num_layers, device=DEVICE)

    # Print the number of parameters
    print("CNN has ", sum(p.numel() for p in cnn.parameters()), " parameters.")
    if rnn_should_be_trained:
        print("RNN has ", sum(p.numel() for p in rnn.parameters()), " parameters.")

    cnn_wrapped = wrap_cnn(cnn, mode="for_spectograms")
    if rnn_should_be_trained:
        rnn_wrapped = wrap_rnn(rnn, mode="for_spectograms")
    # compare_classifiers(dataset=None, model_dic={"cnn": cnn_wrapped, "rnn": rnn_wrapped}, print_summary=True)

    transfer_model_dic_cnn = get_transfer_learning_models_CNN(
        bird_names=["g19o10", "R3428"],
        base_model=cnn,
        arch="CNN",
        wnd_sz=wnd_sz,
        limit=limit,
        retrain_layers=4,
        standardize_input=standardize)

    if rnn_should_be_trained:
        transfer_model_dic_rnn = get_transfer_learning_models_RNN(
            bird_names=["g19o10", "R3428"],
            base_model=rnn,
            arch="RNN",
            wnd_sz=wnd_sz,
            limit=limit,
            network_type=network_type,  # Choose from ["rnn","lstm","gru"]
            hidden_size=hidden_size,
            num_layers=num_layers,
            retrain_layers=4,
            nfreq=128,
            standardize_input=standardize
        )

    transfer_model_dic_cnn["base_CNN"] = cnn
    if rnn_should_be_trained:
        transfer_model_dic_rnn["base_RNN"] = rnn

    for key in transfer_model_dic_cnn:
        transfer_model_dic_cnn[key] = wrap_cnn(transfer_model_dic_cnn[key], mode="for_spectograms", normalize_input=True)

    if rnn_should_be_trained:
        for key in transfer_model_dic_rnn:
            transfer_model_dic_rnn[key] = wrap_rnn(transfer_model_dic_rnn[key], mode="for_spectograms", normalize_input=True)

    compare_classifiers(dataset=None, model_dic=transfer_model_dic_cnn, print_summary=True)
    if rnn_should_be_trained:
        compare_classifiers(dataset=None, model_dic=transfer_model_dic_rnn, print_summary=True)
