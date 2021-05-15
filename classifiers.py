import os.path as path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import dump, load
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    BirdDataLoader, normalize,
    data_path, model_path,
    device, load_bird_data,
    extract_labelled_spectograms,
    train_test_split, create_windows,
    flatten_windows_dic
)

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

def wrap_cnn(cnn, mode = "for_spectograms", normalize_input = True):
    """
    mode has to be one of ["for_spectograms", "for_windows"]
    """
    # Check if the input is just a wrapper function or the real cnn
    if hasattr(cnn,"cnn"):
        real_cnn = cnn.cnn
    else:
        real_cnn = cnn

    def window_wrapper(X):
        if normalize_input:
            X = normalize(torch.tensor(X), mean=real_cnn.mean, std=real_cnn.std)
        X = torch.reshape(X,(-1,1,128,real_cnn.wnd_sz)).to(device)
        pred = real_cnn(X)
        return torch.argmax(pred, dim=1)

    def spectogram_wrapper(spectogram, idx):
        return predict_syllables_CNN(spectogram, idx, window_wrapper, wnd_sz=real_cnn.wnd_sz,
                feature_extraction=real_cnn.feature_extraction)
    
    if mode == "for_windows":
        window_wrapper.is_wrapped = "for_windows"
        window_wrapper.cnn = real_cnn
        return window_wrapper
    elif mode == "for_spectograms":
        spectogram_wrapper.is_wrapped = "for_spectograms"
        spectogram_wrapper.cnn = real_cnn
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
    
    _, peaks_dict = find_peaks(is_on, plateau_size=[5,20000])
    le = peaks_dict['left_edges']
    re = peaks_dict['right_edges']
    return_tuples = [(a,b,idx_spectogram) for a,b in zip(le,re)]
    return return_tuples

def train_CNN(datasets,model_name, feature_extraction=False, normalize_input = True):
    # - Create data loader
    dataloader = BirdDataLoader(datasets['train'], datasets['validation'], datasets['test'], normalize_input = normalize_input)
    
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

    return wrap_cnn(cnn, mode="for_windows", normalize_input = normalize_input)

def get_transfer_learning_models(
    bird_names,
    base_model,
    arch,
    wnd_sz,
    limit,
    retrain_layers=4,
):
    """
    bird_names: list of birdnames in g17y2,g19o10,g4p5,R3428
    base_model: A base model to refine
    arch: str name of the architecture used for storing the model
    wnd_sz: int Window size
    retrain_layers: int Number of layers to be re-trained
    """
    # - Dic to store the models
    base = path.dirname(path.abspath(__file__))
    output_models = {}

    # Load the data and get all labelled spectograms
    bird_data = load_bird_data(names = bird_names)
    data_labelled, _ = extract_labelled_spectograms(bird_data)

    # Perform a train-validation-test split
    data_train, data_test = train_test_split(bird_data = data_labelled, configs = 0.33, seed = 42)
    data_val, data_test = train_test_split(bird_data = data_test, configs = 0.5, seed = 42)

    # Extract the windows from the spectograms
    windows_train, _ = create_windows(bird_data = data_train, wnd_sizes = wnd_sz, limits = limit, on_fracs = 0.5, dt = 5, seed = 42)
    windows_val, _ = create_windows(bird_data = data_val, wnd_sizes = wnd_sz, limits = int(limit/2), on_fracs = 0.5, dt = 5, seed = 42)
    windows_test, _ = create_windows(bird_data = data_test, wnd_sizes = wnd_sz, limits = int(limit/2), on_fracs = 0.5, dt = 5, seed = 42)

    # - Iterate through every bird for every window size
    for bird_name in bird_names:
        print("Start training for bird ", bird_name)
        # - file name
        network_name = "%s_wnd_sz_%s_transfer_bird_%s.model" % (arch,wnd_sz,bird_name)
        network_path = path.join(base, model_path+network_name)

        cnn_transfer = load_cnn(network_path, wnd_sz)
        if cnn_transfer == None:

            X_train, y_train = windows_train[wnd_sz][bird_name]['X'], windows_train[wnd_sz][bird_name]['y']
            X_val, y_val = windows_val[wnd_sz][bird_name]['X'], windows_val[wnd_sz][bird_name]['y']
            X_test, y_test = windows_test[wnd_sz][bird_name]['X'], windows_test[wnd_sz][bird_name]['y']

            # - Create data loader
            dataloader = BirdDataLoader((X_train, y_train), (X_val, y_val), (X_test, y_test))
            
            # - Create data loader test
            data_loader_test = dataloader.get_data_loader(
                dset="test", shuffle=True, num_workers=4, batch_size=64
            )
            cnn_transfer = deepcopy(base_model)
            cnn_transfer.train()
            cnn_transfer.set_data(dataloader.mean, dataloader.std, wnd_sz, False) # use_feature_extraction
            data_loader_train = dataloader.get_data_loader(
                dset="train", shuffle=True, num_workers=4, batch_size=64
            )
            data_loader_val = dataloader.get_data_loader(
                dset="val", shuffle=True, num_workers=4, batch_size=64
            )
            parameters = [p for p in cnn_transfer.children()]
            for parameter in parameters:
                parameter.requires_grad = False
            for parameter in parameters[-retrain_layers:]:
                parameter.requires_grad = True
            print("Retraining ", [x[0] for x in cnn_transfer.named_parameters()][-retrain_layers:])
            optim = torch.optim.Adam(cnn_transfer.parameters(), lr=1e-4)
            n_epochs = 10
            best_val_acc = -np.inf
            for n in range(n_epochs):
                for idx, (data, target) in enumerate(data_loader_train):
                    data, target = data.to(device), target.to(device)  # GPU
                    output = cnn_transfer(data)
                    optim.zero_grad()
                    loss = F.cross_entropy(output, target)
                    if idx % 100 == 0:
                        print(f"Epoch {n} Loss {float(loss)}")
                    loss.backward()
                    optim.step()

                # - Do evaluation
                val_acc = evaluate_model_cnn(cnn_transfer, data_loader_val)
                print(f"Validation accuracy is {val_acc}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(cnn_transfer.state_dict(), network_path)
        
            test_acc = evaluate_model_cnn(cnn_transfer, data_loader_test)
            print(f"Bird {bird_name} Test acc. {test_acc}")
            cnn_transfer.eval()

            # Save the data
            torch.save({"mean": dataloader.mean, 
                        "std": dataloader.std,
                        "wnd_sz": wnd_sz,
                        "feature_extraction": False}, path.join(base,model_path + network_name+".data"))

        output_models[arch+bird_name] = cnn_transfer
    
    return output_models
    

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
        cnn.load_state_dict(torch.load(path, map_location=device))
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

        # This is useful later to check whether there is a wrapper around the rnn
        self.is_wrapped = "not_wrapped"

    def forward(self, x):
        x = x.to(device, dtype=torch.float)
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

def wrap_rnn(rnn, mode = "for_spectograms", normalize_input = True):
    """
    mode has to be one of ["for_spectograms", "for_windows"]
    """
    # Check if the input rnn is the real rnn or just a wrapper function
    if hasattr(rnn,"rnn"):
        real_rnn = rnn.rnn
    else:
        real_rnn = rnn

    def window_wrapper(X):
        if normalize_input:
            X = normalize(torch.tensor(X), mean=real_rnn.mean, std=real_rnn.std)
        X = torch.reshape(X,(-1,128,real_rnn.wnd_sz)).to(device)
        X = torch.transpose(torch.transpose(X,1,2),0,1)
        pred = real_rnn(X)
        return torch.argmax(pred, dim=1)

    def spectogram_wrapper(spectogram, idx):
        return predict_syllables_RNN(spectogram, idx, window_wrapper, wnd_sz=real_rnn.wnd_sz,
                feature_extraction=real_rnn.feature_extraction)
    
    if mode == "for_windows":
        window_wrapper.is_wrapped = "for_windows"
        window_wrapper.rnn = real_rnn
        return window_wrapper
    elif mode == "for_spectograms":
        spectogram_wrapper.is_wrapped = "for_spectograms"
        spectogram_wrapper.rnn = real_rnn
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
    
    _, peaks_dict = find_peaks(is_on, plateau_size=[5,20000])
    le = peaks_dict['left_edges']
    re = peaks_dict['right_edges']
    return_tuples = [(a,b,idx_spectogram) for a,b in zip(le,re)]
    return return_tuples

def train_RNN(datasets,model_name,network_type="rnn", hidden_size = 100, num_layers=1, feature_extraction = False, normalize_input = True):
    
    # Create data loader
    # The parameter 'network_type' always needs to be 'rnn'!
    dataloader = BirdDataLoader(datasets['train'], datasets['validation'], datasets['test'], network_type="rnn", normalize_input = normalize_input) 

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

    return wrap_rnn(rnn, mode="for_windows", normalize_input = normalize_input)

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
        rnn.load_state_dict(torch.load(path, map_location=device))
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

        # This is useful later to check whether there is a wrapper around the rnn
        self.is_wrapped = "not_wrapped"

        if network_type == "rnn":
            self.rnn_cell = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif network_type == "lstm":
            self.rnn_cell = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif network_type == "gru":
            self.rnn_cell = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.output = nn.Linear(in_features=self.hidden_size, out_features=2)

    def forward(self, input):
        input = input.to(device, dtype=torch.float)
        hidden = self.init_hidden(input.shape[1])

        if self.network_type == "lstm":
            rnn_output, hidden_new = self.rnn_cell(input)
            output = self.output(hidden_new[0])
        else:
            rnn_output, hidden_new = self.rnn_cell(input, hidden)
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