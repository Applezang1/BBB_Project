import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci
from scipy.spatial.distance import pdist 

# Set the random seed to have the same random numbers
np.random.seed(0)

# Define the number data examples
n_data = 1000

# Create 1000 data examples (columns) each with 2 dimensions (rows)
n_dim = 2
x_2D = np.random.normal(size=(n_dim,n_data))

# Create 1000 data examples (columns) each with 100 dimensions (rows)
n_dim = 100
x_100D = np.random.normal(size=(n_dim,n_data))

# Create 1000 data examples (columns) each with 1000 dimensions (rows)
n_dim = 1000
x_1000D = np.random.normal(size=(n_dim,n_data))

# Define a function to calculate the ratio between the smallest and largest Euclidean distances between two points
def distance_ratio(x):
  # Compute the smallest Euclidean distance between two points
  smallest_dist = np.min(pdist(x.T, metric='euclidean'))

  # Compute the largest Euclidean distance between two points
  largest_dist = np.max(pdist(x.T, metric='euclidean'))

  # Calculate the ratio and return
  dist_ratio = largest_dist / smallest_dist
  return dist_ratio

# Print the distance ratio given dimensions of the input data
print('Ratio of largest to smallest distance 2D: %3.3f'%(distance_ratio(x_2D)))
print('Ratio of largest to smallest distance 100D: %3.3f'%(distance_ratio(x_100D)))
print('Ratio of largest to smallest distance 1000D: %3.3f'%(distance_ratio(x_1000D)))

# Define a function to calculate the volume of a hypersphere
def volume_of_hypersphere(diameter, dimensions):
  pi = np.pi
  radius = diameter/2
  volume = ((radius**dimensions)*(pi**(dimensions/2)))/(sci.gamma(dimensions/2+1))

  return volume

# Compute the volume of a hypersphere for increasing dimensions of the input data
diameter = 1.0
for c_dim in range(1,11):
  print("Volume of unit diameter hypersphere in %d dimensions is %3.3f"%(c_dim, volume_of_hypersphere(diameter, c_dim)))

# Define a function to compute the proportion of the volume in the outer 1% of the radius
def get_prop_of_volume_in_outer_1_percent(dimension):
  volume = volume_of_hypersphere(diameter, dimension)
  volume99= volume_of_hypersphere(0.99*diameter, dimension)
  proportion = (volume - volume99)/volume
  return proportion

# Compute the volume is in the outer 1% of the radius for increasing dimensions of the input data
for c_dim in [1,2,10,20,50,100,150,200,250,300]:
  print('Proportion of volume in outer 1 percent of radius in %d dimensions =%3.3f'%(c_dim, get_prop_of_volume_in_outer_1_percent(c_dim)))
     