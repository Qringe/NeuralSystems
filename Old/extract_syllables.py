import os.path as path
from scipy.io import loadmat
import os
import numpy as np
from scipy.signal import find_peaks
from utils import *
from joblib import dump, load

def train_syllable_extractor():
    base = path.dirname(path.abspath(__file__))
    download()
    # - Get the training data
    Xs_train = loadmat(path.join(base,"Data/train_dataset/spectrograms/g17y2_train.mat"))["Xs_train"][0]
    annotations = loadmat(path.join(base,"Data/train_dataset/annotations/annotations_g17y2_train.mat"))["annotations_train"][0]

    SETindex = annotations["y"][0]["SETindex"][0][0][0]-1
    ons = annotations["y"][0]["onset_columns"][0][0][0]-1
    offs = annotations["y"][0]["offset_columns"][0][0][0]-1
    ground_truth_tuples = [(a,b,c) for a,b,c in zip(ons,offs,SETindex)]

    print("Max length found", max([b-a for a,b,c in ground_truth_tuples]))
    print("Min length found", min([b-a for a,b,c in ground_truth_tuples]))

    model_path = path.join(base,"Data/syllable_extr.model")
    wnd_sz = 10
    if not path.isfile(model_path):
        X_train, y_train = get_train_data(Xs_train, ground_truth_tuples, wnd_sz=wnd_sz, dt=5, limit=70000)
        clf = train_SVM(X_train,y_train)
        dump(clf, model_path) 
    else:
        clf = load(model_path)

    def syllable_extractor(spectogram, idx):
        return predict_syllables_SVM(spectogram, idx, clf, wnd_sz=wnd_sz)

    return syllable_extractor

def evaluate(syllable_extractor = None):
    base = path.dirname(path.abspath(__file__))
    download()
    # - Get the training data
    Xs_train = loadmat(path.join(base,"Data/train_dataset/spectrograms/g17y2_train.mat"))["Xs_train"][0]
    annotations = loadmat(path.join(base,"Data/train_dataset/annotations/annotations_g17y2_train.mat"))["annotations_train"][0]

    SETindex = annotations["y"][0]["SETindex"][0][0][0]-1
    ons = annotations["y"][0]["onset_columns"][0][0][0]-1
    offs = annotations["y"][0]["offset_columns"][0][0][0]-1
    ground_truth_tuples = [(a,b,c) for a,b,c in zip(ons,offs,SETindex)]

    if syllable_extractor is None:
        syllable_extractor = train_syllable_extractor()

    # - Generate figure for the report
    fig = plt.figure(figsize=(14,6))
    figure_indices = [5,6,7]
    axes = [plt.subplot(len(figure_indices),1,i+1) for i in range(len(figure_indices))]
    for i,idx_spectogram in enumerate(figure_indices):
        for k in axes[i].spines.keys():
            axes[i].spines[k].set_visible(False)
        pred_tuple = syllable_extractor(Xs_train[idx_spectogram],idx_spectogram)
        gt_tuples = [(a,b,c) for (a,b,c) in ground_truth_tuples if c == idx_spectogram]
        plot_spectogram(axes[i], Xs_train[idx_spectogram], pred_tuple, gt_tuples)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_ylabel("Frequency [kHz]")
    axes[0].set_title("Automatic syllable detection")
    if not path.exists(path.join(base,"Data/Figures")):
        os.mkdir(path.join(base,"Data/Figures"))
    plt.savefig(path.join(base,"Data/Figures/syllable_detection.svg"), format='svg')
    plt.show()

    # - Uncomment to run through
    return

    predicted_tuples = []
    correct_length = []
    N = len(np.unique(SETindex))
    for idx_spectogram in np.unique(SETindex):
        print(f"{idx_spectogram}/{N}")
        pred_tuple = syllable_extractor(Xs_train[idx_spectogram],idx_spectogram)
        gt_tuples = [(a,b,c) for (a,b,c) in ground_truth_tuples if c == idx_spectogram]

        if len(pred_tuple) == len(gt_tuples):
            correct_length.append(1)
        else:
            correct_length.append(0)

        plt.clf()
        plot_spectogram(plt.gca(), Xs_train[idx_spectogram], pred_tuple, gt_tuples)
        plt.draw()
        plt.pause(0.2)
        predicted_tuples.extend(pred_tuple)

    print("Percentage of correct tuple lengths ",np.mean(np.array(correct_length)))
    print("Length tuples predicted ",len(predicted_tuples))
    print("Length tuples ground truth ",len(ground_truth_tuples))


if __name__ == "__main__":
    # evaluate()
    use_feature_extraction = False
    wnd_sz = 20
    limit = 70000

    extract_and_store_datasets([20,28,64,104], use_feature_extraction=use_feature_extraction, limit=limit)

    X,y = load_dataset(wnd_sz=wnd_sz, use_feature_extraction=use_feature_extraction, limit=limit)

    cnn = train_CNN(X,y,("CNN_features_%s_wnd_sz_%s_limit_%s.model" % (use_feature_extraction,wnd_sz,limit)))

    def syllable_extractor_CNN(spectogram, idx):
        return predict_syllables_CNN(spectogram, idx, cnn, wnd_sz=wnd_sz,
            feature_extraction=use_feature_extraction)

    evaluate(syllable_extractor=syllable_extractor_CNN)