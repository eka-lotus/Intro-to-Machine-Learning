# Curve Fit and Training Model Project
Author: Samantha Reksosamudra

### Abstract
The goal of this project was to develop and train a machine learning model to predict the best fit model for a given objective function by computing its least-squares error and optimizing the parameters of the fitted model. Three models were used in comparison with each other: linear fit, parabola fit, and 19th degree polynomial fit. All models were trained and tested with a set of data, and linear fit was the best model. The linear fit model provided a minimum error of 8.87% and 4.46% in two different sets of training and test datas.

### Sec. I. Introduction and Overview
Curve fitting is one of the fundamental training models used in machine learning, and there are many training model fits used to predict situations in the real world. So this project explores a couple of algorithm and computational methods to find the best fit for a random dataset that provides the minimum error.

The following code was written in Python and was used to develop a training model for a given objective function as shown below. 
$f(x) = A\cos(Bx) + Cx + D$


### Sec. II. Theoretical Background
Machine learning algorithms are increasingly being used to solve complex real-world problems, including predicting how a mathematical model behaves. In this project, a random dataset of 31 datapoints was given 

```
# Generate some sample data
xdata = np.arange(0,31)
ydata = np.array([30,35,33,32,34,37,39,38,36,36,37,39,42,45,45,41,40,39,42,44,47,49,50,49,46,48,50,53,55,54,53])
```

and we were asked to fit the objective function to the given dataset with least-squares error. The least-squares error method is a mathematical regression analysis technique used to find the best fit line or curve through a set of data points. This method minimizes the sum of squares of the distance between each data point with the fitted line or curve, which the distance is also called as the error. Below is the least-squares error function used in this project.

$E = \sqrt{(1/n)\Sigma_{j=1}^n(f(x_j)-y_j)^2}$

Three fit models were used to create a fit through the datapoints: linear fit, parabola fit, and a 19th degree polynomial fit. The algorithm for linear and parabola fit uses hyperparameter optimization techniques to optimize the model's performance. For the 19th degree polynomial fit, we used a built-in numpy module called the np.polyfit to determine the set of optimized parameters for the model. 

### Sec. III Algorithm Implementation
  ### 2D Loss Landscape of Parameter Pairs
  Before we start training data models, the least-squares error function was used to generate a 2D loss (error) landscape by fixing two parameters and sweeping across the other parameters. Using pcolor, we visualized the loss landscapes in a grid as we sweep through values of different combinations of parameter pairs. This visualization helped to see the local minima of errors as we fix and sweep through different combinations of parameter pairs. 

![objective function image](./obj_function.png)

  ### Train Model Using Training Datapoints
Suppose we use the first 20 datapoints as the first set of training data. 

```
# Get training data from the first 20 data points
td_y = ydata[:20]
td_x = np.arange(0,20)
```

We set up some initial guesses for our parameter to minimize errors, then use opt.minimize to perform opitimization on the given parameters and model fit function. After we store the optimized parameters in a variable, we put the optimized parameters into the model fit function.

```
# Set the initial guess parameter for minimizing error
c1 = np.array([3, 1/3])   

# Define function for fitted line (with 2 constants)
def ls_error_line(c, x, y):
    e2 = np.sqrt(1/xdata.size * np.sum((c[0]*x + c[1]- y)**2))
    return e2

# Perform optimization
res_l= opt.minimize(ls_error_line, c1, args=(td_x, td_y), method='Nelder-Mead')

# Get the optimized parameters
s1 = res_l.x
```

Then, we plot the training datapoints with the model fit with optimized parameters. Below is the first set of training datapoints plotted with the linear fit, parabola fit, and the 19th degree polynomial fit.

![objective function image](./obj_function.png)

Then we repeat the same process, but using the first 10 and last 10 datapoints as the second set of training data for the three model fits. Below is the second set of training datapoints plotted with the linear fit, parabola fit, and the 19th degree polynomial fit.

![objective function image](./obj_function.png)

### Sec. IV. Computational Results
  ### Compute Minimum Error Using Test Datapoints
  Using the remaining 10 datapoints as test data, we tested the training model to those datapoints and compute the minimum error. Below are the least-squares error of the three model fits, with linear fit being the best model fit with least error for the **first** set of training data
```
# Test model to remaining 10 datapoints
test_y1 = ydata[21:31]
test_x1 = np.arange(0,10)

# Compute the error of testing model with the test data

# Minimum error of Fitted Line
error_line = ls_error_line(s1, test_x1, test_y1)
print('Error of linear fit:', error_line)


# Minimum error of Fitted Parabola
error_parabola = ls_error_parabola(s2, test_x1, test_y1)
print('Error of parabola fit:', error_parabola)


# Minimum error of Fitted 19th Degree Polynomial
error_poly = ls_error_poly(c3, test_x1, test_y1)
print('Error of polynomial fit:', error_poly)
```

which prints the following

```
Error of linear fit: 8.872571457046035
Error of parabola fit: 8.8788334860223
Error of polynomial fit: 8.983964598801377
```

And below are the least-squares error of the three model fits, with linear fit being the best model fit with least error for the **second** set of training data

```
Error of linear fit: 4.349457136903958
Error of parabola fit: 4.419167987022885
Error of polynomial fit: 4.761602026870715
```
 
The training model for linear fit provided the best results with minimum errors for both sets of test data. This result shows that **the linear fit provided the best predictions for the given dataset's behavior**, compared with the parabola fit and 19th degree polynomial fit.

### Sec. V. Summary and Conclusions




