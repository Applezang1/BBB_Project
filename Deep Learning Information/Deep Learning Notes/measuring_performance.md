# Chapter 8: Measuring Performance 

## Key Words and Concepts: 
**<ins>Overfitting</ins>**: Overfitting is when a model fits the data too well, fitting the random noise and fluctuations in addition to the data points. This makes it so that the model performs less well on a test set since the model is so accustomed to the training data. 

**<ins>Inductive Bias</ins>**: The tendency/behavior of a model to prioritize one solution over another between data points 

**<ins>Curse of Dimensionality</ins>**: A case where a model with high dimensions overwhelms the number of training data points, leaving lots of gaps between training points

## Possible Sources of Error: 
To measure the performance of a neural network model, we need a test set of input/output pairs. Using the input of the test set, we compute the output using the neural network model and compare the predicted output to the actual output using the loss function to determine the accuracy of the model  

There are three possible sources of error behind why a neural network model might not fit well to the test set, these sources of error are: 

### Noise
**<ins>Noise</ins>**: The model includes noise, which makes it so that there is a possibility of multiple output values (y) for each input value (x). Therefore, the model sometimes makes incorrect assumptions for a certain input value 

Reasoning behind the presence of Noise:  

- The data generation process might be innately random (example: growing two different plants, even though you give the same sunlight and water one might be taller than the other due to genetic factors, can’t control)

- Some of the data is mislabeled 

### Bias
**<ins>Bias</ins>**: Bias is when a model isn’t flexible enough to model the true function correctly. For example, it is impossible to accurately model a parabola with a linear graph. 

### Variance
**<ins>Variance</ins>**: There is a limited amount of training examples and each training example is different from the other, causing variance in model results. The choice of the test set can cause variance in the model’s performance.

## Reducing Error  
In order to reduce the error of the model, the three possible sources of error must be mitigated. The three possible sources of errors can be mitigated in the following method:

**<ins>Noise</ins>**: It is fundamentally impossible to reduce the noise in a neural network model, making noise one of the fundamental limits on the performance of the model 

**<ins>Variance</ins>**: Variance can be reduced by increasing the quantity of the training data to ensure that most of the inputs are well sampled/have output data

**<ins>Bias</ins>**: Bias can be reduced by increasing the model’s capacity (more hidden units/hidden layers) to ensure that the model is flexible enough to model the true function correctly 

### Bias-Variance Trade Off
**<ins>Bias-Variance Trade Off</ins>**: Bias is reduced by increasing the model’s capacity but the variance term typically increases as the model capacity increases. Therefore, neural network models must balance the model’s capacity for optimal model performance 

However, there is a phenomenon named <ins>double descent</ins> where increasing the capacity of the model causes a decrease in the loss until the model has reached its bias-variance trade off point, where the loss then starts to increase. However after increasing the capacity, the loss of the model decreases again even though the variance continues to increase 

- <ins>Classical/under-parameterized regime</ins>: The graphical location of the first decrease in the loss (first descent)

- <ins>Modern/over-parameterized regime</ins>: The graphical location of the second decrease in the loss (second descent)

- <ins>Critical regime</ins>: The graphical location where the loss of the model beings to increase, after the model has passed its bias-variance trade off point 

## Determining Hyperparameters 
In a deep learning model, the capacity of the neural network model depends on the number of hidden layers and hidden units. These two factors are called <ins>hyperparameters</ins>

**<ins>Hyperparameter Search</ins>**: Hyperparameter search involves finding the best hyperparameters that give the lowest loss for a neural network model. 

A hyperparameter search is usually done through training a variety of models with different hyperparameters using a validation set and finding the combination of hyperparameters that resulted in the lowest loss 
