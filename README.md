# Neural Network Implementation

This repository contains a simple implementation of a feedforward neural network with backpropagation, designed to work with the Titanic dataset for binary classification tasks.

## Files

- **`lou.py`**: Contains the `NeuralNetwork` class, which implements the neural network logic, including forward propagation, backpropagation, and training.

## Neural Network Overview

The neural network is implemented in the `lou.py` file and includes the following features:

- **Activation Function**: Sigmoid function.
- **Cost Function**: Mean squared error.
- **Customizable Architecture**: The network structure can be defined by specifying the number of neurons in each layer.

### Key Methods in `NeuralNetwork` Class

1. **`__init__(struct)`**:
   - Initializes the network with random weights and biases based on the provided structure.
   - `struct`: A list of integers representing the number of neurons in each layer.

2. **`frontprop(A0)`**:
   - Performs forward propagation through the network.
   - Returns the activations of each layer.

3. **`train(data, alpha=0.01, tries=10000)`**:
   - Trains the network using backpropagation and gradient descent.
   - `data`: A list of tuples containing input and target output.
   - `alpha`: Learning rate.
   - `tries`: Number of training iterations.

4. **`meancost(data)`**:
   - Computes the mean squared error cost for a given dataset.

## General Neural Network :

This class allows the user to create a neural network with any activation function. 
Because of the general use case of this class the math cannot be optimized for performance.
If you wish for a more performant network use the NeuralNetwork class which has the sigmoid function hard coded in.

## Example

You can find an example of how to use this library in usage_example

## Requirements

- Python 3.8+
- NumPy
- Jupyter Notebook

## License

This project is licensed under the MIT License.
