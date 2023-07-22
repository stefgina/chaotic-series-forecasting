"""
Author: Stefanos Ginargyros
Date: 2023-05-15
Description: EDA, Visualization, Stat Analysis, LGBM Model - Chaotic Series Forecasting
"""


#######################  warnings

import warnings
warnings.filterwarnings( 'ignore' )



#######################  imports

# basic libraries
import random
import pandas    as pd
import numpy     as np
import pickle
import math
from   tqdm   import tqdm

# visualization library
import matplotlib.pyplot   as plt 

# sklearn library
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn  import metrics


###################### randoms

import random
import os
seed = 1983
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)



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
        self.description = 'LGBM - MackeyGlass'
        # input sequence length - look-back
        self.Lag         = 20
        # prediction sequence length
        self.Horizon     = 1
        # target Series
        self.targetSeries = 'MackeyGlass'
        # number of epochs
        self.epochs        = 10000
        # batch size
        self.batch_size    = 1
        # number of workers in DataLoader
        self.num_workers   = 0
        # define verbose
        self.verbose       = True
        # data Log-transformation
        self.Transformation        = False
        # scaling {'Standard', 'MinMax', 'Robust'}
        self.Scaling               = 'Standard'
        # train
        self.train = True
        # infer on which set # ['train', 'val', 'test']
        self.infer_on = 'test' 
        # correlation matrices, euclideans, Coeff etc.
        self.corelation = False
        # scaling active or Not
        self.scaling = False
        # predict Next State
        self.predict = False
        # this is needed for inference debuging mostly
        self.predict2 = False
        # auto-regressively predict the the next autoregressive_samples - lookback based on the first lookback
        self.autoregressive = True
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

parsed_args = parser.parse_args()
args.autoregressive = parsed_args.autoregressive
args.train = parsed_args.train
args.predict = parsed_args.forecast
args.Lag = parsed_args.lookback
args.autoregressive_samples = parsed_args.samples
args.batch_size = parsed_args.batch
args.infer_on = parsed_args.dataset


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






#####################  create dataset


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

train_dataset = create_dataset(train_data, args.Lag)
val_dataset = create_dataset(val_data, args.Lag)
test_dataset = create_dataset(test_data, args.Lag)

print("[INFO] Dataset Shape Incorporating LookBack - Horizon : ",train_dataset[0].shape, train_dataset[1].shape)


trainX = train_dataset[0].squeeze(-1)
trainY = train_dataset[1].squeeze(-1)
# trainY_permut_augmented = np.expand_dims(trainY_permut_augmented, axis=2)

valX = val_dataset[0].squeeze(-1)
valY = val_dataset[1].squeeze(-1)
# valY = np.expand_dims(val_dataset_permut[1], axis=2)

testX = test_dataset[0].squeeze(-1)
testY = test_dataset[1].squeeze(-1)
# testY = np.expand_dims(test_dataset_permut[1], axis=2)\

print("[INFO] Expanded Datasets, Current Shape : ",trainX.shape, trainY.shape)


######################

# Generate random indices for shuffling
indices1 = np.random.permutation(trainX.shape[0])

# Shuffle both arrays using the generated indices
trainX = trainX[indices1]
trainY = trainY[indices1]

# # Generate random indices for shuffling
# indices2 = np.random.permutation(testX.shape[0])

# # Shuffle both arrays using the generated indices
# trainX = testX[indices2]
# trainY = testY[indices2]

######################  create model

import lightgbm as lgb

class LGBMForecast:
    def __init__(self, num_leaves=31, learning_rate=0.1, n_estimators=100):
        self.model = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators)
    
    def train(self, train_data, train_labels):
        self.model.fit(train_data,train_labels)
    
    def forecast(self, test_data):
        predictions = self.model.predict(test_data)
        return predictions


# Create an instance of the LSTM model
model = LGBMForecast()
# model = LGBMForecast (num_leaves=100, learning_rate=0.1, n_estimators=500)
model.train(trainX, trainY)











######################  predictions on test-set 1
pred = None
pred2 = None
data_pred = None
pred_autoregr = None
autoregr_loss = []

if args.predict:
    """
    Perform predictions on the test set.

    """

    # Make predictions
    pred = model.forecast(testX)


######################  autoregressively predict the test-set1

if args.autoregressive:
    """
    Perform autoregressive predictions on the test set.

    """
    # Step 3: Perform autoregressive predictions for the next 20 timesteps
    
    current_input = [testX[0]]

    autoregressive_preds = []
    for _ in tqdm(range(args.autoregressive_samples)):

        next_pred = model.forecast(current_input)
        autoregressive_preds.append(next_pred)
        current_input = np.concatenate((current_input[0][1:], next_pred), axis=0)
        current_input = [current_input]

        print(current_input)
        print(next_pred)

    pred = np.concatenate(autoregressive_preds, axis=0)

    for i in range(pred.shape[0]):

        autoregr_loss.append(metrics.mean_squared_error([testY[i]], [pred[i]]))


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

        setY = testY
        pred = pred
    
        TimeSeriesMackeyGlass[ args.targetSeries ] = setY.flatten()
        TimeSeriesMackeyGlass[ 'Prediction'      ] = pred.flatten()

    if args.autoregressive:

        setY = testY[:args.autoregressive_samples]
        pred = pred[:args.autoregressive_samples]

        print(setY.shape)
        print(pred.shape)
        TimeSeriesMackeyGlass[ args.targetSeries ] = setY.flatten()
        TimeSeriesMackeyGlass[ 'Prediction_Autoregressive' ] = pred.flatten()

    # evaluation
    MAE, RMSE, MAPE, SMAPE, R2, MSE = RegressionEvaluation( TimeSeriesMackeyGlass )
    print('HORIZON:1  \t    MAE %5.4f RMSE %5.4f SMAPE: %5.4f R2: %.4f MSE %5.4f \n' % (MAE, RMSE, SMAPE, R2, MSE) )
    Performance_Foresting_Model = {'RMSE': [], 'MAE': [], 'SMAPE': [], 'R2' : [], 'MSE': []}





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
