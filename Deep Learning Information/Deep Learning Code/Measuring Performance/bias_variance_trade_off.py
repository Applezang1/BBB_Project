import numpy as np
import matplotlib.pyplot as plt

# Define the true function (sin)
def true_function(x):
    y = np.exp(np.sin(x*(2*3.1413)))
    return y

# Generate data points with or without noise
def generate_data(n_data, sigma_y=0.3):
    # Generate x values from [0, 1]
    x = np.ones(n_data)
    for i in range(n_data):
        x[i] = np.random.uniform(i/n_data, (i+1)/n_data, 1)

    # Evaluate the output value using the true function
    # Add noise to the output values
    y = np.ones(n_data)
    for i in range(n_data):
        y[i] = true_function(x[i])
        y[i] += np.random.normal(0, sigma_y, 1)
    return x,y

# Draw the learned/fitted function, together with uncertainty used to generate points
def plot_function(x_func, y_func, x_data=None,y_data=None, x_model = None, y_model =None, sigma_func = None, sigma_model=None):

    fig,ax = plt.subplots()
    ax.plot(x_func, y_func, 'k-')
    if sigma_func is not None:
      ax.fill_between(x_func, y_func-2*sigma_func, y_func+2*sigma_func, color='lightgray')

    if x_data is not None:
        ax.plot(x_data, y_data, 'o', color='#d18362')

    if x_model is not None:
        ax.plot(x_model, y_model, '-', color='#7fe7de')

    if sigma_model is not None:
      ax.fill_between(x_model, y_model-2*sigma_model, y_model+2*sigma_model, color='lightgray')

    ax.set_xlim(0,1)
    ax.set_xlabel('Input, x')
    ax.set_ylabel('Output, y')
    plt.show()

# Compute true function
x_func = np.linspace(0, 1.0, 100)
y_func = true_function(x_func)

# Compute true function but with noise/uncertainty
np.random.seed(1)
sigma_func = 0.3
n_data = 15
x_data,y_data = generate_data(n_data, sigma_func)

# Plot the true function and the function with uncertainty
plot_function(x_func, y_func, x_data, y_data, sigma_func=sigma_func)

# Define neural network model
def network(x, beta, omega):
    # Retrieve number of hidden units
    n_hidden = omega.shape[0]

    y = np.zeros_like(x)
    for c_hidden in range(n_hidden):
        # Evaluate activations for each joint
        line_vals =  x  - c_hidden/n_hidden
        h =  line_vals * (line_vals > 0)
        # Weight activations by omega parameters and sum
        y = y + omega[c_hidden] * h
    # Add bias (beta) and compute output
    y = y + beta

    return y

# Define a function to compute the parameters that minimizes the loss function
def fit_model_closed_form(x,y,n_hidden):
  n_data = len(x)
  A = np.ones((n_data, n_hidden+1))
  for i in range(n_data):
      for j in range(1,n_hidden+1):
          A[i,j] = x[i]-(j-1)/n_hidden
          if A[i,j] < 0:
              A[i,j] = 0

  beta_omega = np.linalg.lstsq(A, y, rcond=None)[0]

  beta = beta_omega[0]
  omega = beta_omega[1:]

  return beta, omega

# Compute the best parameters
beta, omega = fit_model_closed_form(x_data,y_data,n_hidden=3)

# Compute the neural network to output predicted y-value
x_model = np.linspace(0,1,100)
y_model = network(x_model, beta, omega)

# Plot the function and the model
plot_function(x_func, y_func, x_data,y_data, x_model, y_model)

'''Compute the neural network's average mean and variance'''
# Run the model many times with different datasets and return the mean and variance
def get_model_mean_variance(n_data, n_datasets, n_hidden, sigma_func):

  # Create array that stores model results in rows
  y_model_all = np.zeros((n_datasets, x_model.shape[0]))

  for c_dataset in range(n_datasets):
    # Generate n_data x,y, pairs with variance
    x_data,y_data = generate_data(n_data, sigma_func)

    # Compute the best parameters
    beta, omega = fit_model_closed_form(x_data,y_data,n_hidden)

    # Compute the neural network using the best parameters
    y_model = network(x_model, beta, omega)

    # Store the model results
    y_model_all[c_dataset,:] = y_model

  # Compute mean and standard deviation of model
  mean_model = np.mean(y_model_all,axis=0)
  std_model = np.std(y_model_all,axis=0)

  # Return the mean and standard deviation of the fitted model
  return mean_model, std_model

# Define Parameters
n_datasets = 100
n_data = 15
sigma_func = 0.3
n_hidden = 5

# Compute mean and variance of fitted model
np.random.seed(1)
mean_model, std_model = get_model_mean_variance(n_data, n_datasets, n_hidden, sigma_func) 

# Plot the resulting neural network model, with the average mean and variance
plot_function(x_func, y_func, x_model=x_model, y_model=mean_model, sigma_model=std_model)

'''Plot the noise, bias and variance as a function of capacity (# of hidden units)'''
# Define a list of hidden units with increasing number
n_hidden = 12
hidden_variables = list(range(1, n_hidden + 1))

# Define array structure for bias and variance
bias = np.zeros((len(hidden_variables),1)) 
variance = np.zeros((len(hidden_variables),1)) 

# Define neural network model parameters
n_datasets = 100
n_data = 15
sigma_func = 0.3
n_hidden = 5

# Set random seed so that we get the same result every time
np.random.seed(1)

# Compute the variance and bias for each number of hidden units
for c_hidden in range(len(hidden_variables)):
  # Compute the mean and variance of fitted model
  mean_model, std_model = get_model_mean_variance(n_data, n_datasets, hidden_variables[c_hidden], sigma_func) 
  # Compute variance (average squared deviation of fitted models around mean fitted model)
  variance[c_hidden] = np.mean(std_model**2)
  # Compute bias (average squared deviation of mean fitted model around true function)
  bias[c_hidden] = np.mean((mean_model - true_function(x_model))**2)

# Plot the results
fig,ax = plt.subplots()
ax.plot(hidden_variables, variance, 'k-')
ax.plot(hidden_variables, bias, 'r-')
ax.plot(hidden_variables, variance+bias, 'g-')
ax.set_xlim(0,n_hidden)
ax.set_ylim(0,0.5)
ax.set_xlabel("Model capacity")
ax.set_ylabel("Variance")
ax.legend(['Variance', 'Bias', 'Bias + Variance'])
plt.show()
     