import numpy as np
import matplotlib.pyplot as plt

# Define input data (signal)
x = [5.2, 5.3, 5.4, 5.1, 10.1, 10.3, 9.9, 10.3, 3.2, 3.4, 3.3, 3.1]

# Plot the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
plt.show()

# Define a zero-padded convolution operation with a convolution kernel size of 3, a stride of 1, and a dilation of 1
def conv_3_1_1_zp(x_in, omega):
    x_out = np.zeros_like(x_in)
    x_copy = x_in.copy()
    x_copy.insert(0, 0)
    x_copy.insert(len(x_copy), 0)
    output_storage = []
    for i in range(1, len(x_copy)-1):
        output = omega[0]*x_copy[i-1]+ omega[1]*x_copy[i] + omega[2]*x_copy[i+1]
        output_storage.append(output)
    x_out = np.array(output_storage)
    return x_out

# Define weight values
omega = [0.33,0.33,0.33]

# Apply a convolution kernel with kernel size of 3, stride of 1, and dilation of 1
h = conv_3_1_1_zp(x, omega)

# Ensure that the 3, 1, 1 kernel was applied properly
print(f"Sum of output is {np.sum(h):3.3}, should be 71.1")

# Plot the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h, 'r-',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
ax.legend()
plt.show()

# Define weight values
omega = [-0.5,0,0.5]

# Apply a convolution kernel with kernel size of 3, stride of 1, and dilation of 1
h2 = conv_3_1_1_zp(x, omega)

# Plot the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h2, 'r-',label='after')
ax.set_xlim(0,11)
ax.legend()
plt.show()

# Define a zero-padded convolution operation with a convolution kernel size of 3, a stride of 2, and a dilation of 1
def conv_3_2_1_zp(x_in, omega):
    x_out = np.zeros(int(np.ceil(len(x_in)/2)))
    output_storage = []
    x_copy = x_in.copy()
    x_copy.insert(0, 0)
    x_copy.insert(len(x_copy), 0)
    for i in range(1, len(x_copy)-1, 2):
        output = omega[0]*x_copy[i-1]+ omega[1]*x_copy[i] + omega[2]*x_copy[i+1]
        output_storage.append(output)
    x_out = np.array(output_storage)
    return x_out

# Define weight values
omega = [0.33,0.33,0.33]

# Apply a convolution kernel with kernel size of 3, stride of 2, and dilation of 1
h3 = conv_3_2_1_zp(x, omega)

# Ensure that the 3, 2, 1 kernel was defined properly
print(h)
print(h3)

# Define a zero-padded convolution operation with a convolution kernel size of 5, a stride of 1, and a dilation of 1
def conv_5_1_1_zp(x_in, omega):
    x_out = np.zeros_like(x_in)
    output_storage = []
    x_copy = x_in.copy()
    x_copy.insert(0, 0)
    x_copy.insert(len(x_copy), 0)
    x_copy.insert(0, 0)
    x_copy.insert(len(x_copy), 0)
    for i in range(2, len(x_copy)-2, 1):
        output = omega[0]*x_copy[i-2]+ omega[1]*x_copy[i-1] + omega[2]*x_copy[i]+omega[3]*x_copy[i+1] + omega[4]*x_copy[i+2]
        output_storage.append(output)
    x_out = np.array(output_storage)
    return x_out

omega2 = [0.2, 0.2, 0.2, 0.2, 0.2]
h4 = conv_5_1_1_zp(x, omega2)

# Ensure that the 5, 1, 1, kernel was applied correctly 
print(f"Sum of output is {np.sum(h4):3.3}, should be 69.6")

# Plot the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h4, 'r-',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
ax.legend()
plt.show()

# Define a zero-padded convolution operation with a convolution kernel size of 3, a stride of 1, and a dilation of 2
def conv_3_1_2_zp(x_in, omega):
    x_out = np.zeros_like(x_in)
    output_storage = []
    x_copy = x_in.copy()
    x_copy.insert(0, 0)
    x_copy.insert(len(x_copy), 0)
    x_copy.insert(0, 0)
    x_copy.insert(len(x_copy), 0)
    for i in range(2, len(x_copy)-2):
        output = omega[0]*x_copy[i-2]+ omega[1]*x_copy[i] + omega[2]*x_copy[i+2]
        output_storage.append(output)
    x_out = np.array(output_storage)
    return x_out

# Define weight values
omega = [0.33,0.33,0.33]

# Apply a convolution kernel with kernel size of 3, stride of 1, and dilation of 2
h5 = conv_3_1_2_zp(x, omega)

# Ensure that the 3, 1, 2, kernel was computed correctly 
print(f"Sum of output is {np.sum(h5):3.3}, should be 68.3")

# Plot the signal
fig,ax = plt.subplots()
ax.plot(x, 'k-',label='before')
ax.plot(h5, 'r-',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0, 12)
ax.legend()
plt.show()

# Define a function that computes the convolution matrix with kernel size 3, stride 1, and dilation 1 with zero padding
def get_conv_mat_3_1_1_zp(n_out, omega):
  omega_mat = np.zeros((n_out, n_out))
  # For each output position, set the kernel weights at the correct positions
  for i in range(n_out):
    # For kernel size 3, positions are i-1, i, i+1
    for k in range(3):
      col = i + k - 1  # k=0->i-1, k=1->i, k=2->i+1
      if 0 <= col < n_out:
        omega_mat[i, col] = omega[k]
  return omega_mat

# Compute the convolution using the 3, 1, 1 convolution operation
h6 = conv_3_1_1_zp(x, omega)
print(h6)

# Compute the convolution using the convolution matrix
omega_mat = get_conv_mat_3_1_1_zp(len(x), omega)
h7 = np.matmul(omega_mat, x)
print(h7)
