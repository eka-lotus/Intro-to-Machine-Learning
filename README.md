# Curve Fit and Training Model Exercise
Author: Samantha Reksosamudra

### Abstract
The goal of this project was to develop and train a machine learning model to predict the best fit model for a given objective function by computing its least-squares error and optimizing the parameters of the fitted model. Three models were used in comparison with each other: linear fit, parabola fit, and 19th degree polynomial fit. All models were trained and tested with a set of data, and linear fit was the best model. The linear fit model provided a minimum error of 8.87% and 4.46% in two different sets of training and test datas.

### Sec. I. Introduction and Overview
Curve fitting is one of the fundamental training models used in machine learning, and there are many training model fits used to predict situations in the real world. So this project explores a couple of algorithm and computational methods to find the best fit for a random dataset that provides the minimum error.

The following code was written in Python and was used to develop a training model for a given objective function as shown below. 
![objective function image](./obj_function.png)


### Sec. II. Theoretical Background
Machine learning algorithms are increasingly being used to solve complex real-world problems, including predicting how a mathematical model behaves. In this project, a random dataset of 31 datapoints was given 

![objective function image](./obj_function.png)

and we were asked to fit the objective function to the given dataset with least-squares error. The least-squares error method is a mathematical regression analysis technique used to find the best fit line or curve through a set of data points. This method minimizes the sum of squares of the distance between each data point with the fitted line or curve, which the distance is also called as the error. Below is the least-squares error function used in this project.

![objective function image](./obj_function.png)

Three fit models were used to create a fit through the datapoints: linear fit, parabola fit, and a 19th degree polynomial fit. The algorithm for linear and parabola fit uses hyperparameter optimization techniques to optimize the model's performance. For the 19th degree polynomial fit, we used a built-in numpy module called the np.polyfit to determine the set of optimized parameters for the model. 

### Sec. III Algorithm Implementation
  ### 2D Loss Landscape of Parameter
  Before we start training data models, the least-squares error function was used to generate a 2D loss (error) landscape of fixed parameter pairs and 

Suppose we use the first 20 datapoints as the training data. 



### Sec. IV. Computational Results

### Sec. V. Summary and Conclusions




