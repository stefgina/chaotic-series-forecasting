"""
Author: Stefanos Ginargyros
Date: 2023-05-15
Description: EDA, Visualization, Stat Analysis, LSTM Model (Differences * Relative Changes) - Chaotic Series Forecasting
"""

"""
## DRAFT MATH CHECK

Crucial math implemented

- Corellation Coefficient Maths
r = (Σ((x - x̄)(y - ȳ))) / sqrt(Σ(x - x̄)² * Σ(y - ȳ)²)

- Euclidean Distance Maths
Euclidean_distance = sqrt((x_i1 - x_j1)^2 + (x_i2 - x_j2)^2 + ... + (x_id - x_jd)^2)

- Standard Scaler Maths
μ = sum(X) / N, variance = sum((X - μ)^2) / N, σ = sqrt(variance), X_std = (X - μ) / σ

- Stationarity Testing Maths
α + β * t + γ * X(t-1) + δ * ΔX(t-1) + ε(t), where ΔX(t-1) represents the first difference of X(t-1) and ε(t) is the residual or error term.
"""

#######################  warnings

import warnings
warnings.filterwarnings( 'ignore' )



#######################  imports

# basic libraries
import time
import random
import pandas    as pd
import numpy     as np
import pickle
import math
from   tqdm   import tqdm
from scipy.stats import pearsonr

# visualization library
import matplotlib.pyplot   as plt 

# sklearn library
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn  import metrics


# torch libraries
import torch
import torch.nn                     as nn
import torch.nn.functional          as F
from   torch.utils.data             import DataLoader
from   torch.utils.data             import Dataset


###################### randoms

import random
import os
seed = 1983
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True





######################  cuda

try:
    torch.cuda.init()

    if (torch.cuda.is_available() == True):
        print('[INFO] CUDA is available')

        device = torch.device( 'cuda:0' )
        print('[INFO] Device name: %s' % torch.cuda.get_device_name(0))

    else:
        print('[INFO] CUDA is not available')
        device = torch.device( 'cpu' )
except:
    print('[INFO] CUDA is not available')
    device = torch.device( 'cpu' )




######################  parameters

class Parameters():
    """
    Class to hold model parameters.

    Attributes:
    - description (str): Model description.
    - Lag (int): Input sequence length (look-back).
    - Horizon (int): Prediction sequence length.
    - individual (bool): Flag indicating whether to use data through all channels or only individually.
    - enc_in (int): Number of input features.
    - kernel_size (int): Size of the kernel.
    - targetSeries (str): Target series name.
    - epochs (int): Number of epochs.
    - batch_size (int): Batch size.
    - num_workers (int): Number of workers in DataLoader.
    - verbose (bool): Flag indicating whether to print verbose output.
    - learning_rate (float): Learning rate.
    - weight_decay (float): Weight decay.
    - overlap (int): Overlap parameter.
    - clip_grads (bool): Flag indicating whether to clip gradients.
    - clip_grads_norm (float): Norm value for clipping gradients.
    - model_path (str): Path to the trained model.
    - patience (int): Patience parameter.
    - Transformation (bool): Flag indicating whether to perform data log-transformation.
    - Scaling (str): Scaling method ('Standard', 'MinMax', 'Robust').
    - Smoothing (bool): Flag indicating whether to perform data smoothing.
    - window_len (int): Length of the smoothing window.
    - window_type (str): Type of the smoothing window.
    - DataAugmentation (bool): Flag indicating whether to perform data augmentation.
    - visuals (bool): Flag indicating whether to visualize the results.
    - train (bool): Flag indicating whether to train the model.
    - infer_on (str): Dataset to perform inference on ('train', 'val', 'test').
    - corelation (bool): Flag indicating whether to compute correlation matrices, Euclidean distances, coefficients, etc.
    - scaling (bool): Flag indicating whether scaling is active.
    - predict (bool): Flag indicating whether to predict the next state.
    - predict2 (bool): Flag used for inference debugging.
    - autoregressive (bool): Flag indicating whether to auto-regressively predict the whole set.
    - evaluate (bool): Flag indicating whether to evaluate the model using metrics such as MSE, MAE, MAPE, RMSE, R2.
    """

    def __init__(self):

        # model description
        self.description = 'Deep Linear Model - MackeyGlass'
        # input sequence length - look-back
        self.Lag         = 20
        # prediction sequence length
        self.Horizon     = 1
        # data through all channels or only on individual
        self.individual  = True
        # number of input features
        self.enc_in      = 1
        self.kernel_size = 3
        # target Series
        self.targetSeries = 'MackeyGlass'
        # number of epochs
        self.epochs        = 10000
        # batch size
        self.batch_size    = 512
        # number of workers in DataLoader
        self.num_workers   = 0
        # define verbose
        self.verbose       = True
        # learning rate
        self.learning_rate = 1e-4
        # Weight decay
        self.weight_decay  = 0.00001   
        # overlap
        self.overlap = 1     
        # clip gradient
        self.clip_grads      = False
        # norm for clip gradient                
        self.clip_grads_norm = 0.5        
        # trained model path
        self.model_path    = 'LSTM32x32_b512_l20_PROPER_RELATIVE_CHANGES.pth'
        # patience
        self.patience      = 100
        # data Log-transformation
        self.Transformation        = False
        # scaling {'Standard', 'MinMax', 'Robust'}
        self.Scaling               = 'Standard'
        # smoothing
        self.Smoothing             = False
        # for Data Augmentation (parameters to be applied on smoothers)
        self.window_len  = 5    
        self.window_type = 'bartlett' # ['ones', 'hanning', 'hamming', 'bartlett', 'blackman']
        # data Augmentation
        self.DataAugmentation = False
        # differences
        self.differences = False
        # relative_changes
        self.relative_changes = True
        # visualization
        self.visuals = False
        # train
        self.train = False
        # infer on which set # ['train', 'val', 'test']
        self.infer_on = 'test' 
        # correlation matrices, euclideans, Coeff etc.
        self.corelation = False
        # scaling active or Not
        self.scaling = False
        # predict Next State
        self.predict = True
        # this is needed for inference debuging mostly
        self.predict2 = False
        # auto-regressively predict the the next autoregressive_samples - lookback based on the first lookback
        self.autoregressive = False
        self.autoregressive_samples = 1000
        # evaluation Metrics Here - MSE, MAE, MAPE, RMSE, R2
        self.evaluate = True


        # bash combinations

        if self.autoregressive:
            self.batch_size = 1


args = Parameters()

######################  parser
import argparse

parser = argparse.ArgumentParser(description='A simple script')
parser.add_argument('-f', '--forecast', action='store_true', help='Forecast the next state (t+1)')
parser.add_argument('-a', '--autoregressive', action='store_true', help='Autoregressively forecast the next 1000 states')
parser.add_argument('-t', '--train', action='store_true', help='Train the model')
parser.add_argument('-l', '--lookback', help='Lookback length', type=int, default=20)
parser.add_argument('-b', '--batch', help='Batch size', type=int, default=512)
parser.add_argument('-s', '--samples', help='Forecasting samples ahead', type=int, default=1000)
parser.add_argument('-d', '--dataset', help='Dataset to Infer on', type=str, default='test')
parser.add_argument('-m', '--frozenmodel', help='The name of the frozen model', type=str, default='NEW')
parser.add_argument('-e', '--difference', action='store_true', help='Differences in Labels (t2-t1)')
parser.add_argument('-r', '--relative', action='store_true', help='Relative Changes in Labels (t2-t1)/t2')

parsed_args = parser.parse_args()
args.autoregressive = parsed_args.autoregressive
args.train = parsed_args.train
args.predict = parsed_args.forecast
args.Lag = parsed_args.lookback
args.autoregressive_samples = parsed_args.samples
args.batch_size = parsed_args.batch
args.infer_on = parsed_args.dataset
args.model_path = 'models/' + parsed_args.frozenmodel + '.pth'
args.differences = parsed_args.difference
args.relative_changes = parsed_args.relative


# just to be sure we are not cheating
if args.autoregressive:
    args.batch_size = 1


######################  load data

def load_data(file_path):
    """
    Load data from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        dict: The loaded data from the 'data' key of the dictionary.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data['data']

train_data = load_data("./data/MackeyGlass/train.pickle")
val_data = load_data("./data/MackeyGlass/val.pickle")
test_data = load_data("./data/MackeyGlass/test.pickle")

flat_train = train_data.flatten()
flat_val = val_data.flatten()
flat_test = test_data.flatten()

flat_data = np.concatenate((flat_train, flat_val, flat_test))

print("[INFO] Original Dataset Shapes", train_data.shape, val_data.shape, test_data.shape)  


###################### visualization

if args.visuals:
    """
    Visualize data samples from the train, validation, and test datasets.

    - Plots a sample of data points from the train_data, val_data, and test_data arrays.
    - Each array is plotted in a separate subplot using plt.plot.
    """

    # seems like these chunks are continuous....
    pick_sample = 5000
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # plot each array in a separate subplot
    axs[0].plot(flat_train[:pick_sample], color='blue')
    axs[0].set_title('Train data sample')

    axs[1].plot(flat_val[:pick_sample], color='red')
    axs[1].set_title('Validation data sample')

    axs[2].plot(flat_test[:pick_sample], color='green')
    axs[2].set_title('Test data sample')

    # adjust spacing between subplots
    fig.tight_layout()

    plt.show()




######################  transformation

if (args.Transformation == True):

    """
    Apply a logarithmic transformation to the data.

    - Prints a message indicating that the data transformation is applied.
    - Determines the value to be added to the data for the logarithmic transformation.
    - Applies the logarithmic transformation to the train_data, val_data, and test_data arrays using np.log.
    """
    
    print('[INFO] Data transformation applied')
    
    VALUE = np.ceil( max(abs( -flat_data.min().min() ), 1.0) )
    
    train_data  = np.log( train_data + VALUE)
    val_data  = np.log( val_data + VALUE)
    test_data   = np.log( test_data  + VALUE)
    
else:
    print('[INFO] No data transformation applied.')

 





######################  scaling

if args.scaling:
    """
    Apply scaling to the data based on the specified scaling method.

    - Determines the scaling method based on the value of args.Scaling.
    - Applies the selected scaling method to the train_data, val_data, and test_data arrays using the corresponding scaler.
    """
    if (args.Scaling == 'Standard'):
        scaler = StandardScaler()
    elif(args.Scaling == 'MinMax'):
        scaler = MinMaxScaler()
    elif(args.Scaling == 'Robust'):
        scaler = RobustScaler()

    train_data = scaler.fit_transform( train_data.reshape(-1, 1) ).reshape(train_data.shape)
    val_data = scaler.transform( val_data.reshape(-1, 1) ).reshape(val_data.shape) 
    test_data  = scaler.transform( test_data.reshape(-1, 1)).reshape(test_data.shape)

######################  create dataset

train_dataset = None
val_dataset = None
test_dataset = None

train_dataset_og = None
val_dataset_og = None
test_dataset_og = None

trainY_og = None
valY_og = None
testY_og = None


def create_dataset_differences(data, look_back=20):
    """
    Create input-output (difference in targets) pairs from the given data for a time series prediction task.

    Args:
        data (numpy.ndarray): The input data array.
        look_back (int): The number of past time steps to use as input.

    Returns:
        tuple: A tuple containing the input sequences (dataX) and the corresponding output sequences (dataY).

    """
    dataX, dataY = [], []
    for i in range(data.shape[0]):  # for each trajectory
        for j in range(data.shape[1] - look_back):  # for each timestep in the trajectory
            dataX.append(data[i, j:j + look_back])
            dataY.append(data[i, j + look_back]- data[i, j + look_back-1])
    return np.array(dataX), np.array(dataY)

def create_dataset_relative_changes(data, look_back=20):
    """
    Create input-output (difference and division in targets) pairs from the given data for a time series prediction task.

    Args:
        data (numpy.ndarray): The input data array.
        look_back (int): The number of past time steps to use as input.

    Returns:
        tuple: A tuple containing the input sequences (dataX) and the corresponding output sequences (dataY).

    """
    dataX, dataY = [], []
    for i in range(data.shape[0]):  # for each trajectory
        for j in range(data.shape[1] - look_back):  # for each timestep in the trajectory
            dataX.append(data[i, j:j + look_back])
            dataY.append((data[i, j + look_back]- data[i, j + look_back-1])/data[i, j + look_back])
    return np.array(dataX), np.array(dataY)

def create_dataset(data, look_back=20):
    """
    Create input-output pairs from the given data for a time series prediction task.

    Args:
        data (numpy.ndarray): The input data array.
        look_back (int): The number of past time steps to use as input.

    Returns:
        tuple: A tuple containing the input sequences (dataX) and the corresponding output sequences (dataY).

    """
    dataX, dataY = [], []
    for i in range(data.shape[0]):  # for each trajectory
        for j in range(data.shape[1] - look_back):  # for each timestep in the trajectory
            dataX.append(data[i, j:j + look_back])
            dataY.append(data[i, j + look_back])
    return np.array(dataX), np.array(dataY)


if args.differences:
    train_dataset = create_dataset_differences(train_data, args.Lag)
    val_dataset = create_dataset_differences(val_data, args.Lag)
    test_dataset = create_dataset_differences(test_data, args.Lag)

    train_dataset_og = create_dataset(train_data, args.Lag)
    val_dataset_og = create_dataset(val_data, args.Lag)
    test_dataset_og = create_dataset(test_data, args.Lag)

    trainY_og = train_dataset_og[1]
    valY_og = val_dataset_og[1]
    testY_og = test_dataset_og[1]

    print("[INFO] Dataset Shape Incorporating LookBack - Horizon : ",train_dataset[0].shape, train_dataset[1].shape)


if args.relative_changes:
    train_dataset = create_dataset_relative_changes(train_data, args.Lag)
    val_dataset = create_dataset_relative_changes(val_data, args.Lag)
    test_dataset = create_dataset_relative_changes(test_data, args.Lag)

    train_dataset_og = create_dataset(train_data, args.Lag)
    val_dataset_og = create_dataset(val_data, args.Lag)
    test_dataset_og = create_dataset(test_data, args.Lag)

    trainY_og = train_dataset_og[1]
    valY_og = val_dataset_og[1]
    testY_og = test_dataset_og[1]

    print("[INFO] Dataset Shape Incorporating LookBack - Horizon : ",train_dataset[0].shape, train_dataset[1].shape)


else:
    train_dataset = create_dataset(train_data, args.Lag)
    val_dataset = create_dataset(val_data, args.Lag)
    test_dataset = create_dataset(test_data, args.Lag)




######################  create datasets
    
class Data( Dataset ):
    """
    Custom dataset class for training and testing.

    This class inherits from the `torch.utils.data.Dataset` class.

    Args:
        X (numpy.ndarray): The input data array.
        Y (numpy.ndarray): The output data array.

    """
    def __init__(self, X, Y):
        """
        Initialize the Data object.

        Args:
            X (numpy.ndarray): The input data array.
            Y (numpy.ndarray): The output data array.

        """
        self.X    = X
        self.Y    = Y

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.Y)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the input data and the corresponding output data.

        """
        return self.X[ idx ], self.Y[ idx ]


trainX = train_dataset[0]
trainY = train_dataset[1]
# trainY_permut_augmented = np.expand_dims(trainY_permut_augmented, axis=2)

valX = val_dataset[0]
valY = val_dataset[1]
# valY = np.expand_dims(val_dataset_permut[1], axis=2)

testX = test_dataset[0]
testY = test_dataset[1]
# testY = np.expand_dims(test_dataset_permut[1], axis=2)\


print(testX.shape)
print(testY.shape)

# create training and test dataloaders
train_ds = Data(trainX, trainY)
valid_ds = Data(valX, valY)
test_ds  = Data(testX, testY)




##################### dataloaders

# prepare Data-Loaders
train_dl = DataLoader(train_ds, batch_size = args.batch_size, num_workers = args.num_workers, shuffle= True)
valid_dl = DataLoader(valid_ds,  batch_size = args.batch_size, num_workers = args.num_workers)
test_dl  = DataLoader(test_ds,  batch_size = args.batch_size, num_workers = args.num_workers)
print('[INFO] Data loaders were created')

data_batch, _ = next(iter(train_dl))
print("[INFO] Data Batch Shape : ",data_batch.shape)





######################  create model

# Define the LSTM model
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq):
        lstm1_out, _ = self.lstm1(input_seq)
        lstm2_out, _ = self.lstm2(lstm1_out)
        predictions = self.fc(lstm2_out[:, -1, :]) # take the last output of each sequence in the batch
        return predictions
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience = 3, min_delta = 0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                print(f'[INFO] Early stopping')
                return ( True )
            else:
                return ( False )
            
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5, verbose = True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience  = patience
        self.min_lr    = min_lr
        self.factor    = factor
        self.verbose   = verbose
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode      = 'min',
                patience  = self.patience,
                factor    = self.factor,
                min_lr    = self.min_lr,
                verbose   = self.verbose 
            )
        
    def __call__(self, val_loss):
        self.lr_scheduler.step( val_loss )

# Define hyperparameters
input_size = 1
hidden_size = 32
output_size = 1

# Create an instance of the LSTM model
model = LSTMForecast(input_size, hidden_size, output_size)





######################  training parameters

def smape_loss(y_pred, y_true):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) loss. Using this and no the other scipy, or sklearn method because its differentiable directly from torch.

    Args:
        y_pred (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The true values.

    Returns:
        torch.Tensor: The calculated SMAPE loss.

    """
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    diff = torch.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return torch.mean(diff)


# criterion = smape_loss
criterion = nn.MSELoss()
"""
The criterion used for calculating the loss during training.

In this case, it is the Mean Squared Error (MSE) loss from the `nn.MSELoss` class.

"""

optimizer = torch.optim.Adam(params       = model.parameters(), 
                             lr           = args.learning_rate,
                             weight_decay = args.weight_decay)
"""
The optimizer used for training.

In this case, it is the Adam optimizer from the `torch.optim.Adam` class.
It takes the model parameters, learning rate, and weight decay as input.

"""

early_stopping = EarlyStopping(patience  = 50,
                               min_delta = 1.0e-5)

"""
The early stopping object used to stop training when the loss does not improve.

In this case, it is an instance of the `EarlyStopping` class with the specified patience
and minimum delta values.

"""

scheduler = LRScheduler(optimizer = optimizer, 
                        patience  = 10, 
                        min_lr    = 1e-10, 
                        factor    = 0.5, 
                        verbose   = args.verbose)

"""
The learning rate scheduler used to adjust the learning rate during training.

In this case, it is an instance of the `LRScheduler` class with the specified optimizer,
patience, minimum learning rate, factor, and verbosity.

"""




######################  training loop

if args.train:
    """
    Train the model using the provided training data.

    This block performs the training loop for the specified number of epochs.
    It includes forward and backward passes, optimization steps, loss calculation,
    and monitoring of training and validation loss. It also saves the best model
    based on the validation loss and implements early stopping.

    """

    Loss = {
            'Train': [], 
            'Valid':  []
        }    
    """
    Dictionary to store the training and validation loss for each epoch.

    The dictionary has two keys: 'Train' and 'Valid'.
    The values are lists that will store the training and validation loss, respectively.

    """

    for epoch in range(1, args.epochs+1):
        """
        Iterate over the specified number of epochs.

        """

        start = time.time()
        
        train_loss = 0.0
        valid_loss = 0.0    

        batch_idx = 0

        for data, target in train_dl:
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            if (device.type == 'cpu'):
                data   = torch.tensor(data,   dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.float32)
            else:
                data   = torch.tensor(data,   dtype=torch.float32).cuda()
                target = torch.tensor(target, dtype=torch.float32).cuda()

            # outputs = model( data ).squeeze(-1)
            outputs = model(data)
            # calculate the loss
            loss = criterion(outputs, target)
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # clip gradients
            if ( args.clip_grads ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grads_norm)
                
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update running training loss
            train_loss += loss.item()*data.size(0)
                
            # increase batch_idx
            batch_idx  += 1
            
        # print avg training statistics 
        train_loss = train_loss / train_dl.dataset.X.shape[0]

        
        with torch.no_grad():
            """
            Iterate over the validation data.

            """
            for data, target in valid_dl:

                # forward pass: compute predicted outputs by passing inputs to the model
                if (device.type == 'cpu'):
                    data   = torch.tensor(data, dtype=torch.float32)
                    target = torch.tensor(target, dtype=torch.float32)
                else:
                    data   = torch.tensor(data, dtype=torch.float32).cuda()
                    target = torch.tensor(target, dtype=torch.float32).cuda()

                outputs = model(data)
            
                # calculate the loss
                loss = criterion(outputs, target)
                    
                # update running training loss
                valid_loss += loss.item()*data.size(0)
                
        # print avg training statistics 
        valid_loss = valid_loss / valid_dl.dataset.X.shape[0]

        # stop timer
        stop  = time.time()
        
        # show training results
        print('[INFO] Epoch: {:3.0f} Train Loss: {:.6f}\tValid Loss: {:.6f} \tTime: {:.2f}secs'.format(epoch, train_loss, valid_loss, stop-start), end=' ')

        # update best model
        if (epoch == 1):
            Best_score = valid_loss
            
            torch.save(model.state_dict(), args.model_path)
            print("[INFO] Model saved")
        else:
            if (Best_score > valid_loss):
                Best_score = valid_loss
                
                torch.save(model.state_dict(), args.model_path)
                print("[INFO] Model saved")
            else:
                print( )
        
        # store train/val loss
        Loss['Train'] += [ train_loss ]
        Loss['Valid'] += [ valid_loss ]
        
        # learning rate scheduler
        scheduler( valid_loss )
        
        # early Stopping
        if ( early_stopping( valid_loss ) ): break




######################  load optimized model

model.load_state_dict( torch.load( args.model_path ) )
model.eval()

print('[INFO] Model loaded')




######################  inference set decision

if args.infer_on == 'train':
    """
    Set the data loader and data arrays for inference on the training set.

    """
    dl = train_dl
    print("[INFO] Inference on Training-set")
elif args.infer_on == 'valid':
    """
    Set the data loader and data arrays for inference on the validation set.

    """
    dl = valid_dl
    print("[INFO] Inference on Validation-set")
else:
    """
    Set the data loader and data arrays for inference on the test set.

    """
    dl = test_dl
    print("[INFO] Inference on Test-set")





######################  predictions on test-set 1
pred = None
pred2 = None
data_pred = None
pred_autoregr = None
autoregr_loss = None

if args.predict:
    """
    Perform predictions on the test set.

    """

    with torch.no_grad():
        """
        Iterate over the data loader and perform predictions.

        """
        for data, target in tqdm( dl ):

            if (device.type == 'cpu'):
                data   = torch.tensor(data,   dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.float32)
            else:
                data   = torch.tensor(data,   dtype=torch.float32).cuda()
                target = torch.tensor(target, dtype=torch.float32).cuda()
                

            if (pred is None):
                pred = model(data).cpu().detach().numpy()
            else:
                pred = np.concatenate([ pred, model(data).cpu().detach().numpy() ])




    ######################  predictions on test-set 2

    if args.predict2:
        with torch.no_grad():
            for data, target in tqdm( dl ):

                if (device.type == 'cpu'):
                    data   = torch.tensor(data,   dtype=torch.float32)
                    target = torch.tensor(target, dtype=torch.float32)
                else:
                    data   = torch.tensor(data,   dtype=torch.float32).cuda()
                    target = torch.tensor(target, dtype=torch.float32).cuda()
                    

                if (pred2 is None):
                    pred2 = model(data).cpu().detach().numpy()
                else:
                    pred2 = np.concatenate([ pred2, model(data).cpu().detach().numpy()])





        ######################  debugging predictions (optional)

        # define the file path and name
        file_path = 'pred_sorted.txt'
        file_path2 = 'pred2_sorted.txt'

        sorted_pred1 = np.sort(pred, axis=0)
        sorted_pred2 = np.sort(pred2, axis=0)

        print(sorted_pred1)

        print(sorted_pred2)


        # save the array to the text file
        np.savetxt(file_path, sorted_pred1)

        # save the array to the text file
        np.savetxt(file_path2, sorted_pred2)




######################  autoregressively predict the test-set


if args.autoregressive:
    """
    Perform autoregressive predictions on the test set.

    """
    counter = 0

    with torch.no_grad():

        for  data, target in tqdm( dl ):

            """
            Iterate over the data loader and perform autoregressive predictions.

            """
            if (device.type == 'cpu'):
                data   = torch.tensor(data,   dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.float32)
                if counter>0:
                    data_pred = torch.tensor(data_pred[np.newaxis, :, np.newaxis,], dtype=torch.float32)
                
            if (data_pred is None):
                data_pred = data

            if (pred is None):
                pred = model(data_pred).cpu().detach().numpy()

            if (autoregr_loss is None):
                autoregr_loss = [metrics.mean_squared_error(target, pred[-1])]

            else:
                print(pred.shape, model(data_pred).cpu().detach().numpy().shape)
                pred = np.concatenate([ pred, model(data_pred).cpu().detach().numpy() ])
                autoregr_loss.extend([metrics.mean_squared_error(target, pred[-1])])

            data_pred = data_pred.squeeze()
            data_pred = data_pred[1:]
            data_pred = np.concatenate([data_pred, pred[-1]])


            counter += 1

            print(data)
            print(data_pred)

            if counter == args.autoregressive_samples: break




######################  inverse transforms

if args.scaling:
    """
    Apply inverse scaling to the data and predictions.
    """

    # apply inverse scaling
    for i in range( args.Horizon ):
        testX[:,  i]  = scaler.inverse_transform( testX[:,  i].squeeze(-1).reshape(-1,1) )
        valX[:,   i]  = scaler.inverse_transform( valX[:,   i].squeeze(-1).reshape(-1,1) )
        trainX[:, i]  = scaler.inverse_transform( trainX[:, i].squeeze(-1).reshape(-1,1) )

        testY[:,  i]  = scaler.inverse_transform( testY[:,  i].reshape(-1,1) ).squeeze(-1)
        valY[:,   i]  = scaler.inverse_transform( valY[:,   i].reshape(-1,1) ).squeeze(-1)
        trainY[:, i]  = scaler.inverse_transform( trainY[:, i].reshape(-1,1) ).squeeze(-1)

        pred[:, i]    = scaler.inverse_transform( pred[:, i].squeeze(-1).reshape(-1,1) )
        if args.predict2:
            pred2[:, i]   = scaler.inverse_transform( pred2[:, i].squeeze(-1).reshape(-1,1) )


# apply inverse transformation   
if (args.Transformation == True):
    """
    Apply inverse transformation to the data and predictions.
    """
    testX  = np.exp( testX )  - VALUE
    valX   = np.exp( valX )   - VALUE
    trainX = np.exp( trainX ) - VALUE

    testY  = np.exp( testY ) - VALUE
    valY   = np.exp( valY )  - VALUE
    trainY = np.exp( trainY )- VALUE
    pred   = np.exp( pred )  - VALUE
    if args.predict2:
        pred2  = np.exp( pred2 ) - VALUE


if args.differences:
    
    #       dataY.append((data[i, j + look_back]- data[i, j + look_back-1])/data[i, j + look_back])
    #       ENCODED = Y2-Y1/ Y2 -> Y2 = Y1 * (1+ENCODED) -> Y2 = Y1 + Y1*ENCODED 

    # `pred` contains the predicted differences between 21st and 20th values
    # Calculate the original 20th values from the test data
    testX_last_item = testX[:, -1]

    # Convert the differences (`testY`) back to the original 21st values for evaluation

    pred = testX_last_item + pred
    testY = testY_og


if args.relative_changes:

    #       dataY.append((data[i, j + look_back]- data[i, j + look_back-1])/data[i, j + look_back])
    #       ENCODED = Y2-Y1/ Y2 -> Y2 = Y1 * (1+ENCODED) -> Y2 = Y1 + Y1*ENCODED       
    
    # `pred` contains the predicted relative_changes between 21st and 20th values
    # Calculate the original 20th values from the test data
    testX_last_item = testX[:, -1]
    if args.autoregressive:
        testX_last_item = testX[:args.autoregressive_samples, -1]

    # Convert the relative_changes (`testY`) back to the original 21st values for evaluation

    pred = testX_last_item + ( testX_last_item * pred )
    testY = testY_og

######################  inference set decision

if args.infer_on == 'train':
    """
    Set the data loader and data arrays for inference on the training set.

    """
    setY = trainY
    setX = trainX
    print("[INFO] Inference on Training-set")
elif args.infer_on == 'valid':
    """
    Set the data loader and data arrays for inference on the validation set.

    """
    setY = valY
    setX = valX
    print("[INFO] Inference on Validation-set")
else:
    """
    Set the data loader and data arrays for inference on the test set.

    """
    setY = testY
    setX = testX
    print("[INFO] Inference on Test-set")


######################  evaluate performance on test-set

if args.evaluate:
    def smape(A, F):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between the true values (A) and the predicted values (F).

        Args:
            A (np.ndarray): True values.
            F (np.ndarray): Predicted values.

        Returns:
            float: SMAPE score.
        """
        
        try:
            return ( 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) ) )
        except:
            return (np.NaN)
        
    def rmse(A, F):
        """
        Calculate the Root Mean Squared Error (RMSE) between the true values (A) and the predicted values (F).

        Args:
            A (np.ndarray): True values.
            F (np.ndarray): Predicted values.

        Returns:
            float: RMSE score.
        """
        try:
            return math.sqrt(metrics.mean_squared_error(A, F))
        except:
            return (np.NaN)
        
    def mse(A, F):
        """
        Calculate the Mean Squared Error (MSE) between the true values (A) and the predicted values (F).

        Args:
            A (np.ndarray): True values.
            F (np.ndarray): Predicted values.

        Returns:
            float: MSE score.
        """
        try:
            return (metrics.mean_squared_error(A, F))
        except:
            return (np.NaN)


    def RegressionEvaluation( TimeSeriesMackeyGlass ):
        """
        Evaluate the performance of a regression model by calculating various metrics.

        Args:
            TimeSeriesMackeyGlass (pd.DataFrame): DataFrame containing the true values and predicted values.

        Returns:
            tuple: Tuple of performance metrics (MAE, RMSE, MAPE, SMAPE, R2, MSE).
        """
     
        SeriesName = TimeSeriesMackeyGlass.columns[0]
        Prediction = TimeSeriesMackeyGlass.columns[1]
        
        Y    = TimeSeriesMackeyGlass[SeriesName].to_numpy()
        Pred = TimeSeriesMackeyGlass[Prediction].to_numpy()
        
        MAE   = metrics.mean_absolute_error(Y, Pred)
        RMSE  = rmse(Y, Pred)
        MSE = mse(Y, Pred)
        try:
            MAPE  = np.mean(np.abs((Y - Pred) / Y)) * 100.0
        except:
            MAPE  = np.NaN
            
        SMAPE = smape(Y, Pred)
        R2    = metrics.r2_score(Y, Pred)
            
        return (MAE, RMSE, MAPE, SMAPE, R2, MSE)

    print('[INFO] Feature: ', args.targetSeries)
    TimeSeriesMackeyGlass = pd.DataFrame([])        

    if args.predict:
        TimeSeriesMackeyGlass[ args.targetSeries ] = setY.flatten()
        TimeSeriesMackeyGlass[ 'Prediction'      ] = pred.flatten()

    if args.autoregressive:

        setY = setY[:args.autoregressive_samples]
        TimeSeriesMackeyGlass[ args.targetSeries ] = setY.flatten()
        TimeSeriesMackeyGlass[ 'Prediction_Autoregressive' ] = pred.flatten()

    # evaluation
    MAE, RMSE, MAPE, SMAPE, R2, MSE = RegressionEvaluation( TimeSeriesMackeyGlass )
    print('HORIZON:1  \t    MAE %5.4f RMSE %5.4f SMAPE: %5.4f R2: %.4f MSE %5.4f \n' % (MAE, RMSE, SMAPE, R2, MSE) )
    Performance_Foresting_Model = {'RMSE': [], 'MAE': [], 'SMAPE': [], 'R2' : [], 'MSE': []}





######################  plot predictions

if args.visuals:
    subplots = [331, 332, 333, 334, 335, 336,  337, 338, 339]
    plt.figure( figsize = (20, 8) )

    # select random cases
    RandomInstances = [random.randint(1, testY.shape[0]) for i in range(0, 9)]

    for plot_id, i in enumerate(RandomInstances):

        plt.subplot(subplots[plot_id])
        plt.grid()
        plt.plot(i, testY[i], color = 'g', marker = 'o', linewidth = 2)
        plt.plot(i, pred[i],  color = 'r', marker = 'o', linewidth = 2)
        
        plt.legend(['Actual values', 'Prediction'], frameon = False, fontsize = 12)
        plt.ylim([0, 2])
        plt.xticks(rotation = 20)
    plt.show()

if  args.predict:

    # plotting the forecasting of the next state on the whole test set

    plt.plot(setY[:args.autoregressive_samples], 'r', linewidth = 3)  # Plot as red lines
    plt.plot(pred[:args.autoregressive_samples], 'b', linewidth = 1.5)  # Plot as blue dots
    plt.title('Forecast t+1 state - PRED (B) VS LABELS (R)')
    plt.show()

    # Plotting
    plt.scatter(pred, setY, color='c', label='Data points')
    plt.plot(pred, pred, color='r', label='Regression line')

    # Customize plot
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.title('Regression Plot')
    plt.legend()

    # Display plot
    plt.show()


if  args.autoregressive:

    # PREDS VS LABELS
    plt.plot(setY, color='r', linewidth = 3)
    plt.plot(pred, color='b', linewidth = 1.5)
    plt.title(f'Autoregressive Forecast t+{args.autoregressive_samples} states - PRED (B) VS LABELS (R)')
    plt.show()

    #  MSE LOSS of the autoreggresive foreacsting
    plt.plot(autoregr_loss , color='m')
    plt.xlabel('MSE LOSS')
    plt.ylabel('Timesteps')
    plt.title('AUTOREGRESSIVE LOSS MSE')
    plt.show()