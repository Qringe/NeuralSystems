# NeuralSystems
This is the repository for the group project in Neural Systems 2021. This `README` contains the following subsections:
* Requirements
* Setup
* Tasks
* How to use one of the trained models to make predictions
  * Organization and nomenclature of neural networks
  * Making predictions
* How to replicate the results of the report
* How to set up the project to run on the Euler cluster
* Potential errors

## Requirements
All code of this repository has been tested using **Python 3.9.5**. It's very well possible that the code will also run with different Python versions, however, we don't give any guarantees. Additionally to the base python libraries, this project also requires the following libraries:
* joblib==1.0.0
* gdown==3.12.2
* numpy==1.20.3
* scipy==1.6.3
* matplotlib==3.4.2
* sklearn/scikit-learn==0.24.0
* torch==1.8.1

Again, different versions of said libraries might work as well, but we don't give any guarantees.

## Setup
After cloning the github repository, you need to set it up. This can be done by running the setup script:
```bash
python setup.py
```
This will download the training data and ensure that all necessary folders and files are present.

## Tasks
For part C we were asked to solve the following tasks:
* **Task 1**: For each bird, for offline and online variants of the task, and for each of the above window sizes, please train a network that will predict segmentation on the 10% test set of that bird. Please store the optimal hyperparameters (including optimal gap size), optimizer choices, loss functions, and trained network parameters (using pytorch-checkpoint, or a well-documented equivalent method).
* **Task 2**: Can you improve segmentation by extending the pipeline with a filter to remove noise in binary labelling across a given time window? If so, please document your approach and store any parameters that you use.
* **Task 3**: For birds g17y2 and R3428 separately, how much labelled data is needed to have good performance for an entire data set? Characterize the tradeoff between amount of annotation data and improvement in precision and recall of the binary label predictor.
* **Task 4**: Generalization across birds: How well does a segmentation network trained on a subset of the birds perform on a left-out (out-of-distribution) bird?

The names of all trained neural networks are prepended with the name of the task to which they belong.

## How to use a trained model to make predictions
A complete runnable code using the code snippets shown in this tutorial can be found in the file `tutorial_make_predictions.py`.

#### Organization and nomenclature of neural networks
All trained neural networks are stored in the folder `Models`. There are three subfolders in this folder. The folder `Models/All` contains all trained models combined. Our program will look in this folder when loading models. For clarity's sake, all CNN models are also stored in the subfolder `Models/CNN` and all RNN models are also stored in the subfolder `Models/RNN`. The name of each stored neural network is built up of several key-value pairs. The key-value pairs are:
* `task{x}`: This is the first key-value pair of every name. It denotes the task for which this model was trained for (example: `task1_...`)
* `{cnn/rnn}`: This keyword is either `cnn` or `rnn` and unsurprisingly tells whether this trained model is a CNN or an RNN.
* `{g17y2/g19o10/g4p5/R3428}`: This keyword indicates the bird on whose data this model was trained 
* `wnd_{x}`: This key-value pair denotes the window size which was used to train this neural network. 
* `online_{True/False}`: This key-value pair indicates whether the neural network was trained in an online fashion (i.e. using only data from the past for predicting the vocal/non-vocal label of a target column) or an offline fashion. If this parameter is not contained in the name, the network was trained in an offline fashion.
* `size_{x}`: Most of the time it is infeasible to train a neural network using all possible windows which could be extracted from the provided spectograms. This key-value pair denotes the number of windows that were extracted for training this neural network.
* `transfer_learning`: If the name of the neural network contains this keyword, then this model was trained using transfer learning. The base model is always a neural network trained on the bird `g17y2`.
* `tsize{x}`: This key-value pair is only used for transfer-learning models. It denotes the number of windows which were used for transfer learning
* `transfer_bird_{g17y2/g19o10/g4p5/R3428}`: This key-value pair is only used for transfer-learning models. It denotes on which bird the model was trasnfer-learned.

As an example consider the following model name: 

`task4_cnn_transfer_learning_tsize1000__wnd_sz_20_transfer_bird_R3428.model`

From this name we can see that this model is a CNN, it was trained for task 4 and it is a transfer-learning model. It was transfer-learnt on the bird `R3428` using 1000 windows from said bird. The window size of this neural network is 20.

#### Making predictions
In this example we will load a trained CNN and make predictions with it. First choose one of the models in the `Models/All` folder:
```python
cnn_name = "task1_cnn_g17y2_wnd_26_online_False_size_81876"
```
Then you can load the model, using the `load_cnn` command. Besides the path to the model, this function also requires a window size and a boolean parameter telling whether the model was trained in an online fashion or not. This information can be taken from the model name above.
```python
cnn = load_cnn(MODEL_PATH + cnn_name, wnd_sz=26, online=False)
```
Next you need to decide, whether wou want to make predictions for whole spectograms, or for single extracted windows. In either case you need to 'wrap' the CNN using the function `wrap_cnn`:
```python
cnn = wrap_cnn(cnn, mode="for_spectrograms")
```

Choose `mode="for_spectograms"` if you want to make predictions for whole spectograms, and `mode="for_windows"` if you only want to make predictions for single columns.

Now it's time to load the data. There are two options how this can be done:
1. Load the data from wherever and however you like. In this case, just make sure that in the end you have a numpy array containing all spectrograms. I.e. the shape should look like this: 
```python
 [
       # Spectogram 0
       [ [1,2,3,4,5],
         [6,7,8,9,0],
         ...         ],

       # Spectogram 1
       [ [1,2,3,4,5],
         [6,7,8,9,0]
         ...         ],

       ...
    ]
```
2. Use the provided methods. If you plan to use the data which you already gave us for training, there's nothing to do. If you want to add data of a new bird / data which we haven't received for training, you should put a `my_bird_train.mat` file into the `Data/train_dataset/spectrograms` folder and another `annotations_my_bird_train.mat` file into `Data/train_dataset/annotations` folder. Here, `my_bird` is a custom name which you can choose. After this step you should add your custom bird name to the global list storing the names of the bird. This variable is called `BIRD_NAMES` and can be found in the file `utils.py`.

If you chose option 2, this is an example of how to load the data of one or several birds using our provided methods.
```python
bird_data = load_bird_data(names=["g17y2", "g4p5"])

g17y2_spectrograms = bird_data["g17y2"]["Xs_train"]
g4p5_spectrograms = bird_data["g4p5"]["Xs_train"]
```
If the bird data you chose also contains annotations, you can access them as follows:
```python
g17y2_tuples = bird_data["g17y2"]["tuples"]
g4p5_tuples = bird_data["g4p5"]["tuples"]
```
Finally, we can use our loaded CNN to make predictions for the loaded data:
```python
# Iterate over all spectrograms and make predictions for each spectogram
predictions_g17y2 = []
predictions_g4p5 = []

# Make predictions for the first 40 spectrograms of bird g17y2
for index, spectrogram in enumerate(g17y2_spectrograms[:40]):
    predictions_g17y2.extend(
        cnn(g17y2_spectrograms[index], index))

# Make predictions for the first 40 spectrograms of bird g4p5
for index, spectrogram in enumerate(g4p5_spectrograms[:40]):
    predictions_g4p5.extend(
        cnn(g4p5_spectrograms[index], index))
```
The predictions are lists of tuples. Each tuple has the form `(start, end, index)` meaning that the `index`'th spectrogram contains a syllable spanning over the columns in the interval `[start, end]`. As an example, we can print all predicted syllables for spectogram 0 of the bird g17y2:
```python
print([tup for tup in predictions_g17y2 if tup[2] == 0])
# Output: [(26, 53, 0), (214, 259, 0), (445, 489, 0), (640, 685, 0), (844, 888, 0)]
```

## How to replicate the results of the report
This repository contains a Jupyter notebook called `PartC_AutoSyllClust.ipynb` This notebook can be used to reproduce part or all of our results for all 4 tasks. The code can be found below the task statements in the section `Function implementations`. Furthermore, the notebook contains the following cell with global config variables:
```python
run_tasks = {
    "task1": True,
    "task3": True,
    "task4": True,
    "task4_transfer_learning": True
}

# Choose between "Local" and "Euler"
execution_mode = "Euler"
```

The variable `run_tasks` is used to control which results shall be computed. It is a dictionary containing one key for each task (and one additional key for the comparison of the impact of training set size on transfer learning). Each key maps to a boolean variable stating whether the results of this task shall be recomputed.

The second global variable is the `execution_mode` variable. This variable can either take the string `"Local"` or `"Euler"`. This nomenclature is a bit misleading. The `"Local"` execution mode is only used for debugging. It trains only very small neural networks on very small datasets. Like this the whole execution only takes about 5 minutes, so one can quickly check if everything is running alright. The "real" execution mode is `"Euler"`. Technically it's still possible to run the notebook in this execution mode locally on your laptop. However, the total execution time is about 23 hours and running the notebook in this execution mode uses a considerable amount of resources of your computer. Therefore, it is adviced to instead run the notebook on the euler cluster.

As a sidenote, since we used CNNs for all our main results and only experimented with RNNs in the beginning to get a comparison to CNNs, at the moment all lines in the Jupyter notebook concerning the RNN are commented out. You will have to comment them back in if you would like to also reproduce the results of the RNN. Please keep in mind that this step will more or less double the running time of the notebook.

## How to set up the project to run on the Euler cluster.
If you need to know more about the Euler cluster and how to use it than is specified in this subsection, have a look at the official documentation: https://scicomp.ethz.ch/wiki/Getting_started_with_clusters

If you want to replicate our results, it is advisable to run the project on the Euler cluster. This process is straightforward. Once you logged in on the Euler, clone the repository, using the command:
```bash
git clone https://github.com/jubueche/NeuralSystems.git
```
Then you need to set the project up in order to get the training data. This can be done using the setup script:
```bash
python setup.py
```
Since the Euler cluster does not provide a graphical interface, it is not possible to run the Jupyter notebook directly on Euler (at least not without a considerable effort using port-forwarding). An easy solution to this problem is to simply convert the Jupyter notebook to a standard python file. You can achieve this using the following command
```bash
jupyter nbconvert --to script PartC_AutoSyllClust.ipynb
```
Now, feel free to change any of the global variables mentioned in the previous section if you would like to.

In order to run the converted python file on the cluster, we prepared a bash script `run_euler.bash`. You might need to make this script executable first. This can be done using the command:
```bash
chmod u+x run_euler.bash
```
Now you are set up and can simply run the mentioned bash script:
```bash
./run_euler.bash
```
The file `run_euler.bash` uses the following command to submit the proram to the cluster:
```bash
bsub -W 24:00 -n 8 -R rusage[mem=16000,scratch=1000] -N -B -o output_$DATETIME.txt python PartC_AutoSyllClust.py 
```
You can play around with these settings and adjust them to your needs. The parameters are defined as follows:
* `-W` This parameter is used to specify the maximum runtime (hh:mm) which is allocated to run this program. Depending on this number, the program is submitted to a different queue and it might have to wait less/more until it gets started (generally a lower number leads to a shorter waiting time).
* `-n` This parameter is used to specify the number of cores which should be used.
* `-R rusage[mem=x,scratch=y]` This parameter specifies the resource usage. More specifically, it specifies the amount of RAM (in bytes) and the amount of scratch space (in bytes) which should be allocated for each core.
* `-N` If this parameter is set, you will receive an email notification as soon as the execution of your program terminates (either crash or finished successfully).
* `-B` If this parameter is set, you will receive an email notification as soon as the exexcution of your program is started. This can be useful when your job is stuck in the queue for hours.
* `-o` This parameter can be used to specify an output file, to which the terminal output of your program is forwarded to

For a more complete overview over the different submission parameters, have a look at the submission tool: https://scicomp.ethz.ch/lsf_submission_line_advisor/


## Potential errors
* **RuntimeError: CUDA out of memory**: In this case our program tried to run the neural networks on your GPU but either the neural network or its input are too large for the memory of the GPU. In this case you can instruct our program to use the CPU instead. For this, change the variable `DEVICE` in the file `utils.py` to the value `"cpu"`

