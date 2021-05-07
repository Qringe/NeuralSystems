import os.path as path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import dump, load

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils import BirdDataLoader, normalize, data_path, model_path, device

##########################################################################################################################
# Overview of this file: 
## SVM stuff
## CNN stuff
## RNN stuff

##########################################################################################################################
# SVM stuff

def predict_syllables_SVM(X,idx_spectogram,clf,wnd_sz,feature_extraction):
    """
    Take 128 x T sized spectogram and predict the on and off of syllables using SVM
    """
    w = int(wnd_sz / 2)
    is_on = w * [0]
    X_s = []
    for t in range(w,X.shape[1]-w):
        if feature_extraction:
            X_f = extract_features(X[:,t-w:t+w])
        else:
            X_f = X[:,t-w:t+w].reshape((-1))
        X_s.append(X_f)
    X_s = np.array(X_s)
    y_pred = clf.predict(X_s)
    is_on.extend(y_pred)
    is_on.extend(w*[0])
    
    _, peaks_dict = find_peaks(is_on, plateau_size=[5,200])
    le = peaks_dict['left_edges']
    re = peaks_dict['right_edges']
    return_tuples = [(a,b,idx_spectogram) for a,b in zip(le,re)]
    return return_tuples

def train_SVM(X,y,model_name):
    # Check if a model with this name has already been trained:
    base = path.dirname(path.abspath(__file__))
    save_path = path.join(base,model_path+model_name)

    # If such a model exists, load and return it.
    if path.isfile(save_path):
        return load(save_path)

    # Create a train-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Build an SVM model and train it
    print("Training SVM with 2GB of Cache size. Decrease this parameter if it's too much for your computer.")
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',verbose=True,cache_size=2000))
    clf.fit(X_train, y_train)

    # Measure the accuracy of the model on the test set
    y_pred = clf.predict(X_test)
    test_acc = np.mean(np.array(y_pred == y_test, dtype=float))
    print(f"Test acc. is {test_acc}")

    # Store the trained model
    dump(clf, save_path)

    return clf

##########################################################################################################################
# CNN stuff

def wrap_cnn(cnn, mode = "for_spectograms"):
    """
    mode has to be one of ["for_spectograms", "for_windows"]
    """
    def window_wrapper(X):
            X = torch.reshape(normalize(torch.tensor(X), mean=cnn.mean, std=cnn.std),(-1,1,128,cnn.wnd_sz)).to(device)
            pred = cnn(X)
            return torch.argmax(pred, dim=1)

    def spectogram_wrapper(spectogram, idx):
        return predict_syllables_CNN(spectogram, idx, window_wrapper, wnd_sz=cnn.wnd_sz,
                feature_extraction=cnn.feature_extraction)
    
    if mode == "for_windows":
        return window_wrapper
    elif mode == "for_spectograms":
        return spectogram_wrapper
    else:
        raise Exception(f"The mode '{mode}' does not exist! Please choose from ['for_spectograms', 'for_windows']")

def predict_syllables_CNN(X,idx_spectogram,cnn,wnd_sz,feature_extraction):
    """
    Take 128 x T sized spectogram and predict the on and off of syllables using CNN
    """
    w = int(wnd_sz / 2)
    is_on = w * [0]
    X_s = []
    for t in range(w,X.shape[1]-w):
        if feature_extraction:
            X_f = extract_features(X[:,t-w:t+w])
        else:
            X_f = X[:,t-w:t+w]
        X_s.append(X_f)
    X_s = np.array(X_s)
    y_pred = cnn(X_s)
    is_on.extend(y_pred)
    is_on.extend(w*[0])
    
    _, peaks_dict = find_peaks(is_on, plateau_size=[5,200])
    le = peaks_dict['left_edges']
    re = peaks_dict['right_edges']
    return_tuples = [(a,b,idx_spectogram) for a,b in zip(le,re)]
    return return_tuples

def train_CNN(datasets,model_name, feature_extraction=False):
    # - Create data loader
    dataloader = BirdDataLoader(datasets['train'], datasets['validation'], datasets['test'])
    
    # - Create data loader test
    data_loader_test = dataloader.get_data_loader(
        dset="test", shuffle=True, num_workers=4, batch_size=64
    )

    # - Figure out window size
    wnd_sz = dataloader.wnd_sz

    # - Set the seed
    torch.manual_seed(42)

    # - Setup path
    base = path.dirname(path.abspath(__file__))
    save_path = path.join(base,model_path + model_name)

    cnn = load_cnn(save_path, wnd_sz)
    best_val_acc = -np.inf
    if cnn is None:
        cnn = get_CNN_architecture(wnd_sz)
        cnn.set_data(dataloader.mean, dataloader.std, wnd_sz, feature_extraction)

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
            val_acc = evaluate_model_cnn(cnn, data_loader_val)
            print(f"Validation accuracy is {val_acc}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(cnn.state_dict(), save_path)

    test_acc = evaluate_model_cnn(cnn, data_loader_test)
    print(f"Test acc. {test_acc}")
    cnn.eval()

    # Save the data
    torch.save({"mean": dataloader.mean, 
                "std": dataloader.std,
                "wnd_sz": wnd_sz,
                "feature_extraction": feature_extraction}, path.join(base,model_path + model_name+".data"))

    return wrap_cnn(cnn, mode="for_windows")

def evaluate_model_cnn(cnn, data_loader):
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
        data = torch.load(path + ".data")
        cnn.set_data(data['mean'], data['std'], data['wnd_sz'], data['feature_extraction'])
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
        # Changed 10 to 2
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, np.prod(x.shape[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_data(self, mean = None, std = None, wnd_sz = None, feature_extraction = None):
        self.mean = mean
        self.std = std
        self.wnd_sz = wnd_sz
        self.feature_extraction = feature_extraction

def get_CNN_architecture(wnd_sz):
    # - Create sequential model
    cnn = Net(wnd_sz)
    cnn = cnn.to(device)
    return cnn
    
##########################################################################################################################
# RNN stuff

def wrap_rnn(rnn, mode = "for_spectograms"):
    """
    mode has to be one of ["for_spectograms", "for_windows"]
    """
    def window_wrapper(X):
        X = torch.reshape(normalize(torch.tensor(X), mean=rnn.mean, std=rnn.std),(-1,128,rnn.wnd_sz)).to(device)
        X = torch.transpose(torch.transpose(X,1,2),0,1)
        pred = rnn(X)
        return torch.argmax(pred, dim=1)

    def spectogram_wrapper(spectogram, idx):
        return predict_syllables_RNN(spectogram, idx, window_wrapper, wnd_sz=rnn.wnd_sz,
                feature_extraction=rnn.feature_extraction)
    
    if mode == "for_windows":
        return window_wrapper
    elif mode == "for_spectograms":
        return spectogram_wrapper
    else:
        raise Exception(f"The mode '{mode}' does not exist! Please choose from ['for_spectograms', 'for_windows']")


def predict_syllables_RNN(X,idx_spectogram,rnn,wnd_sz,feature_extraction):
    """
    Take 128 x T sized spectogram and predict the on and off of syllables using CNN
    """
    w = int(wnd_sz / 2)
    is_on = w * [0]
    X_s = []
    for t in range(w,X.shape[1]-w):
        if feature_extraction:
            #X_f = extract_features(X[:,t-w:t+w])
            raise Exception("This hasn't been implemented yet for RNNs!")
        else:
            X_f = X[:,t-w:t+w]
        X_s.append(X_f)
    X_s = np.array(X_s)
    y_pred = rnn(X_s)
    is_on.extend(y_pred)
    is_on.extend(w*[0])
    
    _, peaks_dict = find_peaks(is_on, plateau_size=[5,200])
    le = peaks_dict['left_edges']
    re = peaks_dict['right_edges']
    return_tuples = [(a,b,idx_spectogram) for a,b in zip(le,re)]
    return return_tuples

def train_RNN(datasets,model_name,network_type="rnn", hidden_size = 100, num_layers=1, feature_extraction = False):
    
    # Create data loader
    # The parameter 'network_type' always needs to be 'rnn'!
    dataloader = BirdDataLoader(datasets['train'], datasets['validation'], datasets['test'], network_type="rnn") 

    # Create data loader test
    data_loader_test = dataloader.get_data_loader(
        dset="test", shuffle=True, num_workers=4, batch_size=64
    )

    # Figure out window size and the number of rows in the spectogram
    nfreq, wnd_sz = dataloader.nfreq, dataloader.wnd_sz

    # Set the seed
    torch.manual_seed(42)

    # Setup path
    base = path.dirname(path.abspath(__file__))
    save_path = path.join(base,model_path + model_name)

    # Load the rnn if it already exists
    rnn = load_rnn(save_path, network_type = network_type, nfreq = nfreq, 
                hidden_size = hidden_size, num_layers = num_layers, device=device)
    best_val_acc = -np.inf

    # If it doesn't exists, train one
    if rnn is None:

        # Get an untrained rnn
        rnn = get_RNN_architecture(network_type = network_type, nfreq = nfreq, 
                    hidden_size = hidden_size, num_layers = num_layers, device=device)
        
        rnn.set_data(dataloader.mean, dataloader.std, wnd_sz, feature_extraction)

        # Load the training and validation datasets
        data_loader_train = dataloader.get_data_loader(
            dset="train", shuffle=True, num_workers=4, batch_size=64
        )
        data_loader_val = dataloader.get_data_loader(
            dset="val", shuffle=True, num_workers=4, batch_size=64
        )

        # Train the rnn
        optim = torch.optim.Adam(rnn.parameters(), lr=1e-4)
        n_epochs = 10
        for n in range(n_epochs):
            for idx, (data, target) in enumerate(data_loader_train):
                # Adjust input data for RNN
                data = torch.transpose(torch.transpose(data,1,2),0,1)

                # Perform a training step
                output, loss = rnn_train_step(data, target, rnn, optim, device)

                # Print progress
                if idx % 100 == 0:
                    print(f"Epoch {n} Loss {float(loss)}")

            # Do evaluation
            val_acc = evaluate_model_rnn(rnn, data_loader_val)
            print(f"Validation accuracy is {val_acc}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(rnn.state_dict(), save_path)

    test_acc = evaluate_model_rnn(rnn, data_loader_test)
    print(f"Test acc. {test_acc}")
    rnn.eval()

    # Save the data
    torch.save({"mean": dataloader.mean, 
                "std": dataloader.std,
                "wnd_sz": wnd_sz,
                "feature_extraction": feature_extraction}, path.join(base,model_path+model_name+".data"))

    return wrap_rnn(rnn, mode="for_windows")

def rnn_train_step(data, target, rnn, optimizer, device):
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)

    output = rnn(data)

    loss = F.cross_entropy(output, target)
    loss.backward()

    optimizer.step()

    return output, loss.item()

def evaluate_model_rnn(rnn, data_loader):
    rnn.eval()
    correct = 0.0; num = 0.0
    with torch.no_grad():
        for data,target in data_loader:
            data = torch.transpose(torch.transpose(data,1,2),0,1)
            data, target = data.to(device), target.to(device)
            pred = rnn(data)
            correct += torch.sum((torch.argmax(pred,dim=1) == target).float())
            num += len(pred)
    acc = float(correct / num)
    rnn.train()
    return acc

def load_rnn(path, network_type, nfreq = 128, hidden_size=100, num_layers = 1, device="cpu"):
    if os.path.isfile(path):
        rnn = get_RNN_architecture(network_type, nfreq, hidden_size, num_layers, device)
        rnn.load_state_dict(torch.load(path))
        data = torch.load(path + ".data")
        rnn.set_data(data['mean'], data['std'], data['wnd_sz'], data['feature_extraction'])
        rnn.eval()
        return rnn
    else:
        return None

class RNN(nn.Module):
    def __init__(self, network_type, nfreq = 128, hidden_size=100, num_layers = 1, device="cpu"):
        super().__init__()
        if network_type not in ["rnn","lstm","gru"]:
            print("The network type ",network_type," is not recognized. Defaulting to 'rnn'")
            network_type = "rnn"
        
        self.network_type = network_type
        self.nfreq = nfreq
        self.device = device

        self.input_size = nfreq
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if network_type == "rnn":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif network_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif network_type == "gru":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.output = nn.Linear(in_features=self.hidden_size, out_features=2)

    def forward(self, input):
        hidden = self.init_hidden(input.shape[1])

        if self.network_type == "lstm":
            rnn_output, hidden_new = self.rnn(input)
            output = self.output(hidden_new[0])
        else:
            rnn_output, hidden_new = self.rnn(input, hidden)
            output = self.output(hidden_new)

        #print(output.shape)
        #print(output.reshape(-1,2).shape)
        return output[0].reshape(-1,2)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    
    def set_device(self, device):
        if device not in ["cpu","cuda"]:
            raise Exception(f"The device type '{device}' is unknown!")
        self.device = device
        return self.to(device)

    def set_data(self, mean = None, std = None, wnd_sz = None, feature_extraction = None):
        self.mean = mean
        self.std = std
        self.wnd_sz = wnd_sz
        self.feature_extraction = feature_extraction

def get_RNN_architecture(network_type, nfreq = 128, hidden_size=100, num_layers = 1, device="cpu"):
    # Initialize untrained RNN model
    rnn = RNN(network_type, nfreq, hidden_size, num_layers, device)
    rnn = rnn.set_device(device)
    return rnn