# Lorenz Equation and Forecasting Data
Author: Samantha E. Reksosamudra

## Abstract
The goal of this project was to use the Lorenz equation and train some neural networks to advance the solution from $t$ to $\Delta t$ with different values of $\rho$. Then we test on a set of $\rho$ values and compare the forecasting dynamics between the neural networks used in this project. Through the results of least-squared errors computation, it is found that Echo State Networks (ESN) worked the best in forecasting data, followed by Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Feed Forward Neural Network (FFNN). The respective computation results are: 43.13, 50.16, 50.31, and 95.90.
 
## Sec. I. Introduction and Overview
In machine learning, forecasting data refers to the use of statistical models to predict future values of a time series based on historical data. This involves training a machine learning model on past data and using it to generate forecasts of future values. Forecasting data in machine learning can be used in a variety of applications, including finance, healthcare, and weather forecast. This project will evaluate the performances of different neural network architecture used to predict future values of a Lorenz equation.

This project consists of: (1) data preparation by preprocessing, splitting into training and test datasets, and converting them into a format compatible with the neural network, (2) design the model architecture of the feed forward nueral network, (3) train the network using a training algorithm such as backpropagation, (4) evaluate the performance on a test set and generate the error and/or accuracy computation, (5) adjust the parameters, such as learning rate and batch size, by hyperparameter tuning.

## Sec. II. Theoretical Background
A feedforward neural network (FFNN) is a type of artificial neural network that is commonly used in machine learning for supervised learning tasks such as classification and regression. In a feedforward neural network, the information flows in one direction, from the input layer to the output layer, without any feedback loops.

A recurrent neural network (RNN) is a type of neural network that is commonly used in machine learning for applications involving sequential data, such as time series forecasting, speech recognition, and natural language processing. In an RNN, the output of each hidden layer is fed back into the network as input for the next time step, allowing the network to model temporal dependencies in the data. This feedback mechanism allows the network to capture information about the previous states of the sequence and use it to inform its predictions about future states.

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is commonly used in machine learning for applications involving sequential data, such as natural language processing, speech recognition, and time series forecasting. Unlike a traditional RNN, which suffers from the vanishing gradient problem when trying to learn long-term dependencies, an LSTM has a more complex structure that allows it to learn and remember information over longer sequences of data.

An Echo State Network (ESN) is a type of recurrent neural network (RNN) that is used in machine learning for tasks such as time series prediction and signal processing. ESNs are a type of reservoir computing technique, where a large number of randomly connected neurons form a "reservoir" that is used to process the input data. The input is fed into the reservoir, and the output is read out from a subset of the neurons in the reservoir. 

## Sec. III Algorithm Implementation 
  ### Design the Neural Network Architecture
  Using hyperparameter tuning, we created the four model architectures that will be used int this project:
  
  ```
# Create a neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 15)
        self.fc2 = nn.Linear(15, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
  ```
  ```
  # Create an LSTM neural network architecture
class LSTMNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=15, num_layers=1, output_size=3):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        
        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Fully connected layer
        out = self.fc(out[:, :])
        
        return out
  ```
  ```
  # Create an RNN neural network architecture
class RNNNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=15, num_layers=1, output_size=3):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        
        # RNN layer
        out, _ = self.rnn(x, h0)
        
        # Fully connected layer
        out = self.fc(out[:, :])  # Use the last output timestep
        
        return out
  ```
  ```
  # Define the Echo States Network architecture
class ESN(nn.Module):
    def __init__(self, input_size=3, reservoir_size=100, output_size=3):
        super(ESN, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        
        # Reservoir layer
        self.reservoir = nn.Linear(input_size + reservoir_size, reservoir_size)
        
        # Output layer
        self.output = nn.Linear(reservoir_size, output_size)

    def forward(self, x, reservoir_state):
        # Concatenate input with reservoir state
        combined_input = torch.cat([x, reservoir_state], dim=1)
        
        # Reservoir layer
        reservoir_output = torch.tanh(self.reservoir(combined_input))
        
        # Output layer
        output = self.output(reservoir_output)
        
        return output, reservoir_output
  ```
  
 ### Initialize Network and Define Loss Function and Optimizer
 We use Mean Squared Error (MSE) as the loss function and SGD as the optimizer. Below is an example of initializing the FFNN model. If we are training the dataset with another neural network, then we only need to change the initialized network.
 ```
 # Initialize the network and define the loss function and optimizer
  net = Net()
  criterion = nn.MSELoss()  # Mean Squared Error
  optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
 ```
 
 ### Data Preparation for Training Set
 Below are the given set of $\rho$ values to train and test:
 ```
 # Given rho values to train and test
rho_values_train = [10, 28, 40]
rho_values_test = [17, 35]
 ```
 Then we want to define the input and output for all neural networks.
 ```
 # Define the NN input and output
nn_input = np.zeros((100*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)
 ```
  Next, we use the Lorenz equation and perform data preparation tasks, such as tensor transformation, as seen below. The following data preparation technique works for all neural network architecture in this project.
  ```
  # Create a list for input and output for the neural network (after data preparation)
nn_input_final = []
nn_output_final = []

# Data preparation
for rho in rho_values_train:
    
    # Define the Lorenz equation
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                      for x0_j in x0])
    
    for j in range(100):
        nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
        nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
        
    # Convert numpy arrays to PyTorch tensors
    nn_input_tensor = torch.from_numpy(nn_input).float()
    nn_output_tensor = torch.from_numpy(nn_output).float()
    
    # Appending the tensors to a list
    nn_input_final.append(nn_input_tensor)
    nn_output_final.append(nn_output_tensor)
    

# Concatenate the neural network input and outputs from each rho values
nn_in = torch.cat(nn_input_final)
nn_out = torch.cat(nn_output_final)

  ```

 ### Train the Network
 Then we set the number of epochs we want, and run the training set into the algorithm. After it is done, it will evaluate its performance on the test set and compute the least squares error of the dataset.
 ```
# Train the model
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(nn_in)
    loss = criterion(outputs, nn_out)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
 ```
 ### Data Preparation for Testing Set
 Similar to data preparation for training set, but this time, we loop around the $\rho$ test values:
```
# Create a list for input and output for the neural network (for testing)
nn_input_test = []
nn_output_test = []

# Test the network with given test rho values
for rho in rho_values_test:
    
    # Define the Lorenz equation
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                      for x0_j in x0])
    
    for j in range(100):
        nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
        nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
        
    # Convert numpy arrays to PyTorch tensors
    nn_input_tensor = torch.from_numpy(nn_input).float()
    nn_output_tensor = torch.from_numpy(nn_output).float()
    
     # Appending the tensors to a list
    nn_input_test.append(nn_input_tensor)
    nn_output_test.append(nn_output_tensor)

# Concatenate the neural network input and outputs from each rho values
nn_in = torch.cat(nn_input_test)
nn_out = torch.cat(nn_output_test)
```

### Test the Network and Results on Network Performance for Future Predictions 
After we finished preparing the data, we compute the least squares error of the function and see how well the model can predict the future. The results will be used to determine which model has the best forecasting dynamics.
```
# Test the network
with torch.no_grad():
    outputs = model(nn_in)
    compute_mse = criterion(outputs, nn_out)

    print('Least squares error of test data: {}'.format(compute_mse.item()))
```
  
## Sec. IV. Computational Results
  ### Least Squares Error Computation
  The least squares error for FFNN, LSTM, RNN, and ESN are listed as below:
  ```
  # FFNN
  Least squares error of test data: 95.89716339111328
  ``` 
  ```
  # LSTM
  Least squares error of test data: 50.31068801879883
  ```
  ```
  # RNN
  Least squares error of test data: 50.16377258300781
  ```
  ```
  # ESN
  Least squares error of test data: 43.1268310546875
  ```
 
 ### Comparison Between FFNN, LSTM, RNN, and ESN
 It is found that FFNN, LSTM, RNN, and ESN have a least squares error of 95.90, 50.31, 50.16, and 43.13 respectively. It is important to note that the RNN model used here took significantly longer than other models. Specifically, the ESN computation time was one of the most efficient in processing.
 
## Sec. V. Summary and Conclusions
The Echo State Network (ESN) has the best forecasting dynamics compared with the other neural networks used in this project. The computation time also consumed less time, which leads it to become an efficient method for forecasting data.
