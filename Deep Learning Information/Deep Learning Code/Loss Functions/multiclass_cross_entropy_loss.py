import numpy as np
import matplotlib.pyplot as plt

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define a shallow neural network
def shallow_nn(x, beta_0, omega_0, beta_1, omega_1):
    # Ensure that input data is (1 x n_data) array
    n_data = x.size
    x = np.reshape(x,(1,n_data))

    # Compute the hidden unit using matrix multiplication for the input array
    h1 = ReLU(np.matmul(beta_0,np.ones((1,n_data))) + np.matmul(omega_0,x))

    # Compute the output of the shallow neural network using matrix multiplcation for the input array
    model_out = np.matmul(beta_1,np.ones((1,n_data))) + np.matmul(omega_1,h1)
    return model_out

# Define a function to define the parameters
def get_parameters():
  beta_0 = np.zeros((3,1));  # formerly theta_x0
  omega_0 = np.zeros((3,1)); # formerly theta_x1
  beta_1 = np.zeros((3,1));  # three output biases
  omega_1 = np.zeros((3,3)); # nine output weights

  beta_0[0,0] = 0.3; beta_0[1,0] = -1.0; beta_0[2,0] = -0.5
  omega_0[0,0] = -1.0; omega_0[1,0] = 1.8; omega_0[2,0] = 0.65
  beta_1[0,0] = 2.0; beta_1[1,0] = -2; beta_1[2,0] = 0.0
  omega_1[0,0] = -24.0; omega_1[0,1] = -8.0; omega_1[0,2] = 50.0
  omega_1[1,0] = -2.0; omega_1[1,1] = 8.0; omega_1[1,2] = -30.0
  omega_1[2,0] = 16.0; omega_1[2,1] = -8.0; omega_1[2,2] =-8

  return beta_0, omega_0, beta_1, omega_1

# Define a function to plot the data
def plot_multiclass_classification(x_model, out_model, lambda_model, x_data = None, y_data = None, title= None):
  # Format the model data to 1D arrays
  n_data = len(x_model)
  n_class = 3
  x_model = np.squeeze(x_model)
  out_model = np.reshape(out_model, (n_class,n_data))
  lambda_model = np.reshape(lambda_model, (n_class,n_data))

  fig, ax = plt.subplots(1,2)
  fig.set_size_inches(7.0, 3.5)
  fig.tight_layout(pad=3.0)
  ax[0].plot(x_model,out_model[0,:],'r-')
  ax[0].plot(x_model,out_model[1,:],'g-')
  ax[0].plot(x_model,out_model[2,:],'b-')
  ax[0].set_xlabel('Input, x'); ax[0].set_ylabel('Model outputs')
  ax[0].set_xlim([0,1]);ax[0].set_ylim([-4,4])
  if title is not None:
    ax[0].set_title(title)
  ax[1].plot(x_model,lambda_model[0,:],'r-')
  ax[1].plot(x_model,lambda_model[1,:],'g-')
  ax[1].plot(x_model,lambda_model[2,:],'b-')
  ax[1].set_xlabel('Input, x'); ax[1].set_ylabel('lambda or Pr(y=k|x)')
  ax[1].set_xlim([0,1]);ax[1].set_ylim([-0.1,1.05])
  if title is not None:
    ax[1].set_title(title)
  if x_data is not None:
    for i in range(len(x_data)):
      if y_data[i] ==0:
        ax[1].plot(x_data[i],-0.05, 'r.')
      if y_data[i] ==1:
        ax[1].plot(x_data[i],-0.05, 'g.')
      if y_data[i] ==2:
        ax[1].plot(x_data[i],-0.05, 'b.')
  plt.show()

'''Softmax Function'''
# Define the Softmax Function
def softmax(model_out):
  # Compute the exponential of the model outputs
  exp_model_out = np.exp(model_out)

  # Compute the sum of the exponentials 
  sum_exp_model_out = np.sum(exp_model_out, axis = 0, keepdims=True)

  # Normalize the exponentials 
  softmax_model_out = exp_model_out/sum_exp_model_out

  return softmax_model_out

# 1D training data, input/output pairs
x_train = np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train = np.array([2,0,1,2,1,0,\
                    0,2,2,0,2,0,\
                    2,0,1,2,1,2, \
                    1,0])

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Define input values
x_model = np.arange(0,1,0.01)

# Compute the shallow neural network
model_out= shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)

# Pass the output of the shallow neural network through a softmax function
lambda_model = softmax(model_out)

# Plot the model and how it matches with the training data
plot_multiclass_classification(x_model, model_out, lambda_model, x_train, y_train)

# Define a function to return the probability of each output value
def categorical_distribution(y, lambda_param):
    return np.array([lambda_param[row, i] for i, row in enumerate (y)])

# Here are three examples
print(categorical_distribution(np.array([[0]]),np.array([[0.2],[0.5],[0.3]])))
print(categorical_distribution(np.array([[1]]),np.array([[0.2],[0.5],[0.3]])))
print(categorical_distribution(np.array([[2]]),np.array([[0.2],[0.5],[0.3]])))

'''Likelihood Function (Categorical Distribution)'''
def compute_likelihood(y_train, lambda_param):
  likelihood = np.prod(categorical_distribution(y_train, lambda_param))

  return likelihood

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Compute parameters of the categorical distribution using a shallow neural network
model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Pass the output of the shallow neural network through a softmax function
lambda_train = softmax(model_out)

# Compute the likelihood function of the cateogiral distribution
likelihood = compute_likelihood(y_train, lambda_train)

# Ensure that the likelihood function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(0.000000041,likelihood))

'''Negative Log Likelihood Function (Categorical Distribution)'''
def compute_negative_log_likelihood(y_train, lambda_param):
  nll = -np.sum(np.log(categorical_distribution(y_train, lambda_param)))

  return nll

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Compute the parameters of the categorical distribution using a shallow neural network
model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Pass the outputs of the shallow neural network through the softmax function
lambda_train = softmax(model_out)

# Compute the negative log likelihood of the categorical distribution
nll = compute_negative_log_likelihood(y_train, lambda_train)

# Ensure that the negative log likelihood function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(17.015457867,nll))

'''Minimizing the Loss Function by optimizing beta_1'''
# Define a range of values for the parameter (beta_1)
beta_1_vals = np.arange(-2,6.0,0.1)

# Define array structure for likelihoods and negative log likelihoods
likelihoods = np.zeros_like(beta_1_vals)
nlls = np.zeros_like(beta_1_vals)

beta_0, omega_0, beta_1, omega_1 = get_parameters()
for count in range(len(beta_1_vals)):
  # Set the value for the parameter
  beta_1[0,0] = beta_1_vals[count]

  # Compute the network with new parameters
  model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)
  lambda_train = softmax(model_out)

  # Compute and store the likelihood and the negative log likelihood
  likelihoods[count] = compute_likelihood(y_train,lambda_train)
  nlls[count] = compute_negative_log_likelihood(y_train, lambda_train)

  # Plot the model for every 20th parameter setting
  if count % 20 == 0:
    # Run the model to get values to plot and plot it.
    model_out = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
    lambda_model = softmax(model_out)
    plot_multiclass_classification(x_model, model_out, lambda_model, x_train, y_train, title="beta1[0,0]=%3.3f"%(beta_1[0,0]))

# Plot the likelihood and negative log likelihood for each beta_1 value
fig, ax = plt.subplots()
fig.tight_layout(pad=5.0)
likelihood_color = 'tab:red'
nll_color = 'tab:blue'


ax.set_xlabel('beta_1[0, 0]')
ax.set_ylabel('likelihood', color = likelihood_color)
ax.plot(beta_1_vals, likelihoods, color = likelihood_color)
ax.tick_params(axis='y', labelcolor=likelihood_color)

ax1 = ax.twinx()
ax1.plot(beta_1_vals, nlls, color = nll_color)
ax1.set_ylabel('negative log likelihood', color = nll_color)
ax1.tick_params(axis='y', labelcolor = nll_color)

plt.axvline(x = beta_1_vals[np.argmax(likelihoods)], linestyle='dotted')

plt.show()

# Print the maximum likelihood and minimum negative log likelihood for the best beta_1 value
print("Maximum likelihood = %f, at beta_1=%3.3f"%( (likelihoods[np.argmax(likelihoods)],beta_1_vals[np.argmax(likelihoods)])))
print("Minimum negative log likelihood = %f, at beta_1=%3.3f"%( (nlls[np.argmin(nlls)],beta_1_vals[np.argmin(nlls)])))

# Plot the best model
beta_1[0,0] = beta_1_vals[np.argmin(nlls)]
model_out = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
lambda_model = softmax(model_out)
plot_multiclass_classification(x_model, model_out, lambda_model, x_train, y_train, title="beta1[0,0]=%3.3f"%(beta_1[0,0]))

     
     
