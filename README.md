# Sensors and Decoders
Author: Samantha E. Reksosamudra

## Abstract
The goal of this project was to do analysis on the performance of SHallow REcurrent Decoders (SHRED) from a given code found here: https://github.com/shervinsahba/pyshred. We trained the model, plot results and do an analysis of the performance as a function of time lag variable, Gaussian noise, and number of sensors. Time lag and number of sensors variable have a positive relationship with the loss value, while Gaussian noise increases the loss.  

## Sec. I. Introduction and Overview
In practical applications, LSTM networks are used for tasks such as sentiment analysis, speech recognition, handwriting recognition, and predicting future values in time series data. Decoders, on the other hand, are employed to generate human-readable outputs, such as generating coherent sentences in natural language processing tasks or reconstructing images from encoded representations in computer vision tasks.

The code in the given repository (by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz) provides an overview of the SHallow REcurrent Decoders (SHRED) model that "learn a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state." And the dataset used to train and test this model includes sea-surface temperature (SST), a forced turbulent flow, and atmospheric ozone concentration.

This project consists of: (1) an example usage found in ```example.ipynb```, (2) define the values to train the model on based on the variable we're testing (i.e. time lag, Gaussian noise, number of sensors), (3) train the network using the SHRED algorithm, (4) evaluate the performance on a test set and generate the loss values, and (5) repeat for other variables to test. Additionally, we can adjust the parameters, such as learning rate, epoch size, and batch size, by hyperparameter tuning.

## Sec. II. Theoretical Background
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is designed to process and learn from sequential data. It is particularly effective in handling long-term dependencies and capturing patterns over time. LSTMs use a memory cell and various gates (input, forget, and output) to control the flow of information and update the memory cell state. They are widely used in applications such as natural language processing (NLP), speech recognition, time series analysis, and machine translation.

In the context of machine learning, decoders refer to a component or a network architecture responsible for generating output from learned representations or features. They take encoded or hidden representations from an encoder network (which captures relevant information from input data) and transform them into the desired output format. Decoders are commonly used in tasks such as language generation, image captioning, machine translation, and text-to-speech synthesis.


## Sec. III Algorithm Implementation 
  ### Train Model
  Based on the example code, we prepared the input data and trained the model using the SST dataset.
  ```
  import numpy as np
from processdata import load_data, TimeSeriesDataset
import models
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Randomly select 3 sensor locations 
num_sensors = 3 

# Set the trajectory length (lags) to 52, corresponding to one year of measurements.
lags = 52
 
# Load the data
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]

sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

### Train the network
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=30, lr=1e-3, verbose=True, patience=5)
  ```
  
  ### Test Model
  Then we calculated the loss value of this model which prints a value of 0.039674558.
  
  ```
  test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
  ```
  
  ### Plot Results
  ```
  from processdata import load_full_SST

# SST data with world map indices for plotting
full_SST, sst_locs = load_full_SST()
full_test_truth = full_SST[test_indices, :]

# replacing SST data with our reconstruction
full_test_recon = full_test_truth.copy()
full_test_recon[:,sst_locs] = test_recons

# reshaping to 2d frames
for x in [full_test_truth, full_test_recon]:
    x.resize(len(x),180,360)
    
    plotdata = [full_test_truth, full_test_recon]
labels = ['truth','recon']
fig, ax = plt.subplots(1,2,constrained_layout=True,sharey=True)
for axis,p,label in zip(ax, plotdata, labels):
    axis.imshow(p[0])
    axis.set_aspect('equal')
    axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)
  ```
  
  
 ### Data Preparation for a Performance Analysis
 Firstly, we define the values of the variable we want to test.
 
 ```
 # Define time lag values to test
time_lag_values = [1, 10, 28, 34, 47, 52, 65, 78, 84, 99]

# Define variables of number of sensors to test
num_sensors = [3, 5, 9, 15, 20, 55, 67, 80, 100]

# Define the parameters for Gaussian noise
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Specify the range of noise (standard deviation) levels 
mean = 0  # Mean of the Gaussian noise
 ```
  
## Sec. IV. Computational Results
  ### abc
  
## Sec. V. Summary and Conclusions


