import os.path as path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# - Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

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
            

def get_train_data(spectograms, ground_truth_tuples, wnd_sz, limit=10000, dt=3, use_features=True):
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
        if use_features:
            X_tmp = extract_features(spectograms[idx][:,l:r])
        else:
            X_tmp = spectograms[idx][:,l:r]
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

def train_SVM(X,y):
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
        os.system("unzip Data/train_dataset.zip -d Data/")

def normalize(X, mean=None, std=None):
    if not (mean == None):
        return (X - mean) / std
    else:
        return (X - torch.mean(X)) / torch.std(X)

class BirdDataLoader:

    def __init__(self,X,y):
        self.wnd_sz = X.shape[2]
        # - Split data 
        self.X_train, X_test, self.y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        self.mean = torch.mean(torch.tensor(self.X_train).float())
        self.std = torch.std(torch.tensor(self.X_train).float())

        self.train_dataset = TensorDataset(torch.reshape(normalize(torch.tensor(self.X_train).float()),(-1,1,128,self.wnd_sz)),torch.tensor(self.y_train))
        self.val_dataset = TensorDataset(torch.reshape(normalize(torch.tensor(self.X_val).float()),(-1,1,128,self.wnd_sz)),torch.tensor(self.y_val))
        self.test_dataset = TensorDataset(torch.reshape(normalize(torch.tensor(self.X_test).float()),(-1,1,128,self.wnd_sz)),torch.tensor(self.y_test))

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

def train_CNN(X,y,model_name):
    # - Create data loader
    dataloader = BirdDataLoader(X,y)

    # - Create data loader test
    data_loader_test = dataloader.get_data_loader(
        dset="test", shuffle=True, num_workers=4, batch_size=64
    )

    # - Figure out window size
    wnd_sz = X.shape[2]

    # - Set the seed
    torch.manual_seed(42)

    # - Setup path
    base = path.dirname(path.abspath(__file__))
    save_path = path.join(base,"Data/"+model_name)

    cnn = load_cnn(save_path, wnd_sz)
    best_val_acc = -np.inf
    if cnn is None:
        cnn = get_CNN_architecture(wnd_sz)
        data_loader_train = dataloader.get_data_loader(
            dset="train", shuffle=True, num_workers=4, batch_size=64
        )
        data_loader_val = dataloader.get_data_loader(
            dset="val", shuffle=True, num_workers=4, batch_size=64
        )
        optim = torch.optim.Adam(cnn.parameters(), lr=1e-4)
        n_epochs = 10
        for n in range(n_epochs):
            for idx, (data, target) in enumerate(data_loader_train):
                data, target = data.to(device), target.to(device)  # GPU
                output = cnn(data)
                optim.zero_grad()
                loss = F.cross_entropy(output, target)
                if idx % 100 == 0:
                    print(f"Epoch {n} Loss {float(loss)}")
                loss.backward()
                optim.step()

            # - Do evaluation
            val_acc = evaluate_model(cnn, data_loader_val)
            print(f"Validation accuracy is {val_acc}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(cnn.state_dict(), save_path)

    test_acc = evaluate_model(cnn, data_loader_test)
    print(f"Test acc. {test_acc}")
    cnn.eval()

    def model_wrapper(X):
        X = torch.reshape(normalize(torch.tensor(X), mean=dataloader.mean, std=dataloader.std),(-1,1,128,wnd_sz)).to(device)
        pred = cnn(X)
        return torch.argmax(pred, dim=1)

    return model_wrapper

def evaluate_model(cnn, data_loader):
    cnn.eval()
    correct = 0.0; num = 0.0
    for data,target in data_loader:
        data, target = data.to(device), target.to(device)  # GPU
        pred = cnn(data)
        correct += torch.sum((torch.argmax(pred,dim=1) == target).float())
        num += len(pred)
    acc = float(correct / num)
    cnn.train()
    return acc

def load_cnn(path, wnd_sz):
    if os.path.isfile(path):
        cnn = get_CNN_architecture(wnd_sz)
        cnn.load_state_dict(torch.load(path))
        cnn.eval()
        return cnn
    else:
        return None

def compute_output_dim(M,N,K0,K1,P0=0,P1=0,S0=1,S1=1,D0=1,D1=1):
    HOUT = int((M + 2*P0 - D0*(K0-1)-1) / S0 +1)
    WOUT = int((N + 2*P1 - D1*(K1-1)-1) / S1 +1)
    return HOUT,WOUT

class Net(nn.Module):
    def __init__(self, wnd_sz):
        super().__init__()
        self.wnd_sz = wnd_sz
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        HOUT,WOUT = compute_output_dim(128,self.wnd_sz,5,5)
        HOUT = int(HOUT / 2)
        WOUT = int(WOUT / 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        HOUT2,WOUT2 = compute_output_dim(HOUT,WOUT,5,5)
        HOUT2 = int(HOUT2 / 2)
        WOUT2 = int(WOUT2 / 2)
        self.fc1 = nn.Linear(16 * HOUT2 * WOUT2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, np.prod(x.shape[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_CNN_architecture(wnd_sz):
    # - Create sequential model
    cnn = Net(wnd_sz)
    cnn = cnn.to(device)
    return cnn