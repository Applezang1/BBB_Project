# Chapter 10: Convolutional Networks 

## Convolutional Neural Network 
**<ins>Convolutional Neural Networks</ins>** are a type of network that consists of convolutional layers and is mainly used for image processing.  

**<ins>Convolutional Layers</ins>** are used in CNN, which use parameters shared across the whole image instead of independent parameters, allowing the model to learn local patterns of the overall image 

### Covariant 
A CNN is translation-equivariant, which means that any translation based transformation to the input data transforms the output data in a predictable way. This is useful in images, where moving a cat 10 pixels to the left will also move the CNN’s activation map 10 pixels to the left 

### Computation of CNN 
In convolutional neural networks, the output is predicted by taking the weighted sum of the previous, current, and subsequent input values.

#### Problem: 

When CNN is at the first input or last input value, there is no previous or subsequent input value 

#### Solution: 

- **<ins>Zero-padding</ins>** assumes that any input values that’s outside of the range takes the value of zero

- **<ins>Valid Convolutions</ins>** discards any output values where there is no previous or subsequent input values to compute the output 

### Convolving the Input
The equation for the weighted sum of input values to predict the output (convolutional kernel) can be changed through the following three parameters where: 

- <ins>Stride</ins> changes the movement of the current input values, where a stride of two means that the current input value jumps from x_1 to x_3

- <ins>Size</ins> changes the amount of input values that contributes to the output, where a size of 5 means that 5 input values centered around the central input value are used to compute one output 

- <ins>Dilation</ins> changes the distance between the input values, where a dilation of 2 means that the input values are separated by a distance of 2. (so input value is used every other time) 

This methodology of computing the CNN is called **<ins>convolving the input</ins>**

A **<ins>convolutional layer</ins>** takes the computed output through convolving the input and adds a bias term and puts the resulting output into an activation equation 

### Channels 
Often, convolution takes place multiple times for better results. This works by: 

Convolving the input and using it to create a convolutional layer. Each convolution creates a unique set of hidden units, which are called <ins>channels</ins>

### Receptive Field 
<ins>Receptive fields</ins> are regions of the original data that affect the hidden units. CNNs have multiple receptive fields, which allows hidden units to capture larger portions of the original data 

### 2D Convolutional Networks 
2D convolutional networks are very similar to 1D convolutional networks but take a 2D object as the convolutional kernel rather than a 1D object. 

## Downsampling 
Downsampling is an approach that scales down the spatial dimensions (length and width) of the feature maps (all the hidden units in each channel) of a convolutional network. For a 2x2 region in a feature map, this is done through: 

- **<ins>Max pooling</ins>**, which retains the maximum of the hidden units in the 2x2 feature map 

- **<ins>Mean pooling</ins>**, which retains the average of the hidden units in the 2x2 feature map

- **<ins>Subsampling</ins>**, which picks the hidden unit in a specific/certain location 

## Upsampling 
Upsampling is an approach that scales up the spatial dimensions (length and width) of the feature maps (all the hidden units in each channel) of a convolutional network. For a 2x2 region in a feature map, this is done through:  

- Duplicating all the hidden units  

- Reversing the effects of max pooling  

- Duplicating the hidden units and filling up the intermediate spots by interpolating the values 

**<ins>Transposed convolution</ins>**, where each input value contributes to the values of multiple hidden units and therefore increases the spatial dimensions of the feature maps 

### Function of CNN
Convolution networks are used in semantic segmentation (assigning a label to each pixel in an image), object detection, and image classification 
