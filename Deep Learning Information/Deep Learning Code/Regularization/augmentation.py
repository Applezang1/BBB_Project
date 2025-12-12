import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import mnist1d
import random

# Import the Training Dataset
args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args, path='./mnist1d_data.pkl', download=False, regenerate=False)

# The training and test input and outputs are in
# data['x'], data['y'], data['x_test'], and data['y_test']
print("Examples in training set: {}".format(len(data['y'])))
print("Examples in test set: {}".format(len(data['y_test'])))
print("Length of each example: {}".format(data['x'].shape[-1]))

D_i = 40    # Input dimensions
D_k = 200   # Hidden dimensions
D_o = 10    # Output dimensions

# Define a model with two hidden layers with 200 hidden units each
model = nn.Sequential(
nn.Linear(D_i, D_k),
nn.ReLU(),
nn.Linear(D_k, D_k),
nn.ReLU(),
nn.Linear(D_k, D_o))

# Define a function that initializes the weight values (parameter)
def weights_init(layer_in):
  # Initialize the parameters with He initialization
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

# Define the cross entropy loss function as the chosen loss function
loss_function = torch.nn.CrossEntropyLoss()

# Define the SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)

# Define a variable that decreases learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Convert training and test data into proper format for training
x_train = torch.tensor(data['x'].astype('float32'))
y_train = torch.tensor(data['y'].transpose().astype('long'))
x_test= torch.tensor(data['x_test'].astype('float32'))
y_test = torch.tensor(data['y_test'].astype('long'))

# Load the converted data into a class that creates the batches
data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

# Initialize model weights
model.apply(weights_init)

# Define the number of times to loop over the dataset 
n_epoch = 50

# Define array structure for the loss and the % correct at each epoch
errors_train = np.zeros((n_epoch))
errors_test = np.zeros((n_epoch))

# Undergo back propagation for each loop over the dataset and calculate statistics
for epoch in range(n_epoch):
  # Loop over batches
  for i, batch in enumerate(data_loader):
    # Retrieve inputs and labels for the batch
    x_batch, y_batch = batch
    # Reset the parameter gradients to zero
    optimizer.zero_grad()
    # Calculate forward pass and model output
    pred = model(x_batch)
    # Compute the loss
    loss = loss_function(pred, y_batch)
    # Compute backward pass
    loss.backward()
    # Undergo the SGD 
    optimizer.step()

  # Compute statistics for this epoch
  pred_train = model(x_train)
  pred_test = model(x_test)
  _, predicted_train_class = torch.max(pred_train.data, 1)
  _, predicted_test_class = torch.max(pred_test.data, 1)
  errors_train[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
  errors_test[epoch]= 100 - 100 * (predicted_test_class == y_test).float().sum() / len(y_test)
  print(f'Epoch {epoch:5d}, train error {errors_train[epoch]:3.2f}, test error {errors_test[epoch]:3.2f}')

# Plot the test error and training error of the neural network model
fig, ax = plt.subplots()
ax.plot(errors_train,'r-',label='train')
ax.plot(errors_test,'b-',label='test')
ax.set_ylim(0,100); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
ax.set_title('Train Error %3.2f, Test Error %3.2f'%(errors_train[-1],errors_test[-1]))
ax.legend()
plt.show()

# Define a function to augment (to make subsets of) the main dataset
def augment(input_vector):
  # Define array structure for output
  data_out = np.zeros_like(input_vector)

  # Offset the data by a random integer
  data_out = np.roll(input_vector, random.randint(1, len(input_vector) - 1))

  # Randomly scale each data point by a factor randomly chosen from [0.8, 1.2]
  data_out = np.random.uniform(0.8, 1.2, size = data_out.shape)*data_out

  return data_out

# Define an array to store the original data
n_data_orig = data['x'].shape[0]

# Double the amount of data
n_data_augment = n_data_orig+4000

# Create array structures to store the augmented data
augmented_x = np.zeros((n_data_augment, D_i))
augmented_y = np.zeros(n_data_augment)

# Store the original data in the augmented data array structure
augmented_x[0:n_data_orig,:] = data['x']
augmented_y[0:n_data_orig] = data['y']

# Loop over dataset to augment each data point
for c_augment in range(n_data_orig, n_data_augment):
  # Choose a random data point
  random_data_index = random.randint(0, n_data_orig-1)
  # Augment the point and store it in the augmented data array
  augmented_x[c_augment,:] = augment(data['x'][random_data_index,:])
  augmented_y[c_augment] = data['y'][random_data_index]

# Define the cross entropy loss as the loss function
loss_function = torch.nn.CrossEntropyLoss()

# Define the SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)

# Define a variable that decreases learning rate by half every 50 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Convert training and test data into proper format for training
x_train = torch.tensor(augmented_x.astype('float32'))
y_train = torch.tensor(augmented_y.transpose().astype('long'))
x_test= torch.tensor(data['x_test'].astype('float32'))
y_test = torch.tensor(data['y_test'].astype('long'))

# Load the converted data into a class that creates the batches
data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

# Initialize model weights
model.apply(weights_init)

# Define the number of times to loop over the entire dataset
n_epoch = 50

# Define array structures to store the loss and the % correct at each epoch
errors_train_aug = np.zeros((n_epoch))
errors_test_aug = np.zeros((n_epoch))

# Undergo back propagation for each loop over the dataset and calculate statistics
for epoch in range(n_epoch):
  # Loop over batches
  for i, batch in enumerate(data_loader):
    # Retrieve inputs and labels for the batch
    x_batch, y_batch = batch
    # Reset the parameter gradients to zero
    optimizer.zero_grad()
    # Compute the forward pass and the model output
    pred = model(x_batch)
    # Compute the loss
    loss = loss_function(pred, y_batch)
    # Compute the backward pass
    loss.backward()
    # Undergo the SGD
    optimizer.step()

  # Compute statistics for this epoch
  pred_train = model(x_train)
  pred_test = model(x_test)
  _, predicted_train_class = torch.max(pred_train.data, 1)
  _, predicted_test_class = torch.max(pred_test.data, 1)
  errors_train_aug[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
  errors_test_aug[epoch]= 100 - 100 * (predicted_test_class == y_test).float().sum() / len(y_test)
  print(f'Epoch {epoch:5d}, train error {errors_train_aug[epoch]:3.2f}, test error {errors_test_aug[epoch]:3.2f}')

# Plot the test error and training error of the neural network model
fig, ax = plt.subplots()
ax.plot(errors_train,'r-',label='train')
ax.plot(errors_test,'b-',label='test')
ax.plot(errors_test_aug,'g-',label='test (augmented)')
ax.set_ylim(0,100); ax.set_xlim(0,n_epoch)
ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
ax.set_title('TrainError %3.2f, Test Error %3.2f'%(errors_train_aug[-1],errors_test_aug[-1]))
ax.legend()
plt.show()
     