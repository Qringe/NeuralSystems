import os.path as path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def reverse_on_off(tuples,T):
    reversed_tuples =  []
    next_on = 0
    if tuples == []:
        return []
    for on,off,idx in tuples:
        reversed_tuples.append((next_on,on,idx))
        next_on = off
    reversed_tuples.append((next_on,T-1,idx))
    reversed_tuples = [(a,b,c) for (a,b,c) in reversed_tuples if not a == b]
    return reversed_tuples
            

def get_train_data(spectograms, ground_truth_tuples, wnd_sz, limit=10000, dt=3):
    """
    Extract 128 x wnd_sz chunks from within and outside [on,off] and store in list.
    Extract features from the list and store in array. Return array.
    """
    Xs = []; ys = []
    w = int(wnd_sz / 2)
    last_idx = 0
    current_pool = []
    n_added_total = 0
    def get_wnd_data(t,idx,):
        l = int(t-w)
        r = int(t+w)
        if l < 0 or r >= spectograms[idx].shape[1]:
            return None
        X_tmp = extract_features(spectograms[idx][:,l:r])
        return X_tmp

    for idx_tuple,tup in enumerate(ground_truth_tuples):
        print(f"Data {idx_tuple}/{len(ground_truth_tuples)}")
        if tup[2] == last_idx:
            current_pool.append(tup)
        else:
            n_added = 0
            for on,off,idx in current_pool:
                for t in range(int(on),int(off),dt):
                    X_f = get_wnd_data(t, idx)
                    if X_f is None:
                        continue
                    Xs.append(X_f)
                    ys.append(1)
                    n_added += 1
                    n_added_total += 1
            for on,off,idx in reverse_on_off(current_pool, spectograms[last_idx].shape[1]):
                for t in range(int(on),int(off),dt):
                    if n_added == 0:
                        break
                    X_f = get_wnd_data(t, idx)
                    if X_f is None:
                        continue
                    Xs.append(X_f)
                    ys.append(0)
                    n_added -= 1
                    n_added_total += 1

            if n_added_total >= limit:
                break

            last_idx = tup[2]
            current_pool = []

    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs[:limit],ys[:limit]

def train_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = np.mean(np.array(y_pred == y_test, dtype=float))
    print(f"Test acc. is {test_acc}")
    return clf

def extract_features(X):
    X = np.array(X, dtype=np.float32)
    return [np.mean(X), np.std(X), np.median(X), np.mean(np.abs(X)), np.max(X)-np.min(X)]

def plot_spectogram(ax, X, tuples_predicted, tuples_gt):
    maxfreq = 10
    nonoverlap = 128
    fs = 32000
    ax.imshow(np.flip(X, axis = 0))
    xt = np.arange(X.shape[1]*nonoverlap/fs)
    for t_p in tuples_predicted:
        ax.add_patch(matplotlib.patches.Rectangle((t_p[0], 0), t_p[1]-t_p[0], 15,linewidth=2,
                edgecolor='r',
                facecolor='r',                                     
                fill = True)
        )
    for t_gt in tuples_gt:
        ax.add_patch(matplotlib.patches.Rectangle((t_gt[0], 30), t_gt[1]-t_gt[0], 15,linewidth=2,
                edgecolor='g',
                facecolor='g',                                     
                fill = True)
        )

def download():
    base = path.dirname(path.abspath(__file__))
    p = path.join(base,"Data/train_dataset.zip")
    if not path.isfile(p):
        # pip install gdown
        os.system("gdown https://drive.google.com/uc?id=1nvLex6f5HY48eZ6MyvVcP122Q5N9sDr2")
        os.system("mv train_dataset.zip Data/")
        os.system("unzip train_dataset.zip")