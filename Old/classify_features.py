import os.path as path
import numpy as np
from scipy.io import loadmat
from joblib import dump, load

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

from extract_features import create_features
from utils import download

def list_errors(Xs_features, X_test, y_test, y_pred):
    """
    Creates for each possible error the list of spectograms where this error occured.
    For example, it creates one list of all spectograms, where 'B' syllables have been classified as 'C'
    This is useful for manual inspection of said spectograms.
    """
    # Download the data if it doesn't exist yet
    base = path.dirname(path.abspath(__file__))
    download()

    # Load some data
    annotations = loadmat(path.join(base,"Data/train_dataset/annotations/annotations_g17y2_train.mat"))["annotations_train"][0]
    SETindex = annotations["y"][0]["SETindex"][0][0][0]-1

    # Hash all arrays in order to enable lookups
    Xs_features_h = [hash(array.tobytes()) for array in Xs_features]
    X_test_h = np.array([hash(array.tobytes()) for array in X_test])

    # A dictionary to convert the class labels to letters
    dic = {2: 'A', 3: 'B', 4: 'C', 5: 'D', 7: 'E', 9: 'F'}
    y_test, y_pred = np.array(y_test), np.array(y_pred)

    spec_indices = []
    # Group all errors into their different classes
    for c1 in dic.keys():
        for c2 in dic.keys():
            if c1 == c2:
                continue
            # All entries of X_test where y_test == c1 and y_pred == c2
            X_test_wrong = X_test_h[
                np.intersect1d(
                    np.where(y_test == c1),
                    np.where(y_pred == c2)
                )]
            
            # All entries of Xs_features where y_test == c1 and y_pred == c2
            indices = sorted([SETindex[Xs_features_h.index(i)] for i in X_test_wrong])
            if indices == []:
                continue
            spec_indices.append((dic[c1],dic[c2],indices))
    
    return spec_indices


def get_labels():
    """
    Loads the cluster labels and returns them
    """
    # Download the data if it doens't exist yet
    base = path.dirname(path.abspath(__file__))
    download()

    # Load the syllable labels
    annotations = loadmat(path.join(base,"Data/train_dataset/annotations/annotations_g17y2_train.mat"))["annotations_train"][0]
    return annotations["y"][0]["clust"][0][0][0]

def train_syllable_classifier():
    # Check if the extracted features are already present
    base = path.dirname(path.abspath(__file__))
    data_path = path.join(base,"Data/Xs_features.data")
    if not path.isfile(data_path):
        # If not, create the features
        create_features()

    # Load the syllable features and the syllable labels
    Xs_features = load(data_path)
    clust = get_labels()

    # Create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(Xs_features,clust)

    # A simple SVM
    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # clf.fit(X_train, y_train)

    # A gradient boosting classifier
    model_params = {'n_estimators' : 100, 'num_leaves' : 50, 'max_depth' : 10,  
                        'n_jobs' : -1, 'silent' : True}
    clf = LGBMClassifier(**model_params).fit(X_train,y_train)

    # Check how well the trained model predicts the test set
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    
    print(f"The syllable classifier achieves an accuracy of {round(100 * score,2)}% on the test set.")
    print("Confusion matrix:\n",confusion_matrix(y_test,y_pred))

    # Print the ids of all spectograms where errors occured
    errors = list_errors(Xs_features, X_test, y_test,y_pred)
    print("Spectograms in which the classifier made errors:")
    for c1, c2, indices in errors:
        print(f"true: {c1}, pred: {c2}, list = {indices}")


if __name__ == "__main__":
    train_syllable_classifier()