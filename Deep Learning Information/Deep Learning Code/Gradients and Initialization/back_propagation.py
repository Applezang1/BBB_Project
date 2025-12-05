import numpy as np
import matplotlib.pyplot as plt

# Set seed in order to consistently obtain the same random number
np.random.seed(0)

# Number of hidden layers
K = 5
# Number of neurons per layer
D = 6
# Input layer
D_i = 1
# Output layer
D_o = 1

# Construct an empty list for the weights and biases
all_weights = [None] * (K+1)
all_biases = [None] * (K+1)

# Construct an array structure and initialize the weights and biases of the input and output layers
all_weights[0] = np.random.normal(size=(D, D_i))
all_weights[-1] = np.random.normal(size=(D_o, D))
all_biases[0] = np.random.normal(size =(D,1))
all_biases[-1]= np.random.normal(size =(D_o,1))

# Construct intermediate hidden layers
for layer in range(1,K):
  all_weights[layer] = np.random.normal(size=(D,D))
  all_biases[layer] = np.random.normal(size=(D,1))

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define a function to compute a neural network
def compute_network_output(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights) -1
  
  # Store the pre-activations (all_f) and the activations (all_h)
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  all_h[0] = net_input

  # Calculate the pre-activation and activation for each hidden layer
  for layer in range(K):
      # Compute the pre-activation function
      all_f[layer] = all_biases[layer] + np.matmul(all_weights[layer], all_h[layer])
      # Compute the activation function
      all_h[layer+1] = ReLU(all_f[layer])

  # Compute the output of the neural network from the last hidden layer
  all_f[K] = all_biases[K] + np.matmul(all_weights[K], all_h[K])

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

# Define Input Values
net_input = np.ones((D_i,1)) * 1.2

# Compute Network Output
net_output, all_f, all_h = compute_network_output(net_input,all_weights, all_biases)
print("True output = %3.3f, Your answer = %3.3f"%(1.907, net_output[0,0]))

# Define the Least Squares Loss Function
def least_squares_loss(net_output, y):
  return np.sum((net_output-y) * (net_output-y))

# Define the Derivative of the Least Squares Loss Function (in respect to net_output)
def d_loss_d_output(net_output, y):
    return 2*(net_output -y)

# Define Output Values
y = np.ones((D_o,1)) * 20.0

# Compute the loss of the neural network
loss = least_squares_loss(net_output, y)
print("y = %3.3f Loss = %3.3f"%(y, loss))

# Define an Indicator Function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>0] = 1
  x_in[x_in<=0] = 0
  return x_in

'''Backward Pass'''
# Define a function for the main backward pass
def backward_pass(all_weights, all_biases, all_f, all_h, y):
  # Store the derivatives dl_dweights and dl_dbiases in lists
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # Store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)

  # Compute derivatives of the loss with respect to the network output
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))

  # Compute the backward pass
  for layer in range(K,-1,-1):
    # Calculate the derivatives of the loss with respect to the biases at layer
    all_dl_dbiases[layer] = all_dl_df[layer]

    # Calculate the derivatives of the loss with respect to the weights at layer
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].T)

    # Calculate the derivatives of the loss with respect to the activations
    all_dl_dh[layer] = np.matmul(all_weights[layer].T, all_dl_df[layer])

    if layer > 0:
      # Calculate the derivatives of the loss with respect to the pre-activation f
      all_dl_df[layer-1] = indicator_function(all_f[layer-1])*np.matmul(all_weights[layer].T, all_dl_df[layer])

  return all_dl_dweights, all_dl_dbiases

# Compute the Backward Pass
all_dl_dweights, all_dl_dbiases = backward_pass(all_weights, all_biases, all_f, all_h, y)

np.set_printoptions(precision=3)

'''Compute Derivatives with Finite Differences'''
# Define a list for derivatives computed by finite differences
all_dl_dweights_fd = [None] * (K+1)
all_dl_dbiases_fd = [None] * (K+1)

# Ensure that the derivatives were calculated correctly using finite differences
delta_fd = 0.000001

# Test the dervatives of the bias vectors
for layer in range(K+1):
  dl_dbias  = np.zeros_like(all_dl_dbiases[layer])
  # For every element in the bias
  for row in range(all_biases[layer].shape[0]):
    # Take copy of biases and change one element at a time
    all_biases_copy = [np.array(x) for x in all_biases]
    all_biases_copy[layer][row] += delta_fd
    network_output_1, *_ = compute_network_output(net_input, all_weights, all_biases_copy)
    network_output_2, *_ = compute_network_output(net_input, all_weights, all_biases)
    dl_dbias[row] = (least_squares_loss(network_output_1, y) - least_squares_loss(network_output_2,y))/delta_fd
  all_dl_dbiases_fd[layer] = np.array(dl_dbias)
  print("-----------------------------------------------")
  print("Bias %d, derivatives from backprop:"%(layer))
  print(all_dl_dbiases[layer])
  print("Bias %d, derivatives from finite differences"%(layer))
  print(all_dl_dbiases_fd[layer])
  if np.allclose(all_dl_dbiases_fd[layer],all_dl_dbiases[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
    print("Success!  Derivatives match.")
  else:
    print("Failure!  Derivatives different.")

# Test the derivatives of the weights matrices
for layer in range(K+1):
  dl_dweight  = np.zeros_like(all_dl_dweights[layer])
  # For every element in the bias
  for row in range(all_weights[layer].shape[0]):
    for col in range(all_weights[layer].shape[1]):
      # Take copy of biases and change one element at a time
      all_weights_copy = [np.array(x) for x in all_weights]
      all_weights_copy[layer][row][col] += delta_fd
      network_output_1, *_ = compute_network_output(net_input, all_weights_copy, all_biases)
      network_output_2, *_ = compute_network_output(net_input, all_weights, all_biases)
      dl_dweight[row][col] = (least_squares_loss(network_output_1, y) - least_squares_loss(network_output_2,y))/delta_fd
  all_dl_dweights_fd[layer] = np.array(dl_dweight)
  print("-----------------------------------------------")
  print("Weight %d, derivatives from backprop:"%(layer))
  print(all_dl_dweights[layer])
  print("Weight %d, derivatives from finite differences"%(layer))
  print(all_dl_dweights_fd[layer])
  if np.allclose(all_dl_dweights_fd[layer],all_dl_dweights[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
    print("Success!  Derivatives match.")
  else:
    print("Failure!  Derivatives different.")
     
     

     