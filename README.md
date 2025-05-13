# Neural Network Implementation

This repository contains a simple implementation of a feedforward neural network with backpropagation, designed to work with the Titanic dataset for binary classification tasks.

## Files

- **`lou.py`**: Contains the `NeuralNetwork` class, which implements the neural network logic, including forward propagation, backpropagation, and training.
- **`titanicdataset.ipynb`**: A Jupyter Notebook that preprocesses the Titanic dataset, trains the neural network, and generates predictions for submission.
- **`titanic/`**: Contains the Titanic dataset files:
  - `train.csv`: Training data.
  - `test.csv`: Test data.
  - `gender_submission.csv`: Example submission file.
- **`submission.csv`**: The output file containing predictions for the test dataset.

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

## Titanic Dataset Preprocessing

The `titanicdataset.ipynb` notebook includes the following preprocessing steps:

1. **Data Cleaning**:
   - Dropping irrelevant columns (`Name`, `Ticket`, `Cabin`, `PassengerId`).
   - Filling missing values in the `Age` column with the mean.
   - Mapping categorical values (`Sex` and `Embarked`) to numerical values.
   - Removing rows with any remaining missing values.

2. **Normalization**:
   - Normalizes numerical columns (`Age`, `Pclass`, `Parch`, `Fare`) to the range [0, 1] using a custom `normalizer` function.

3. **Data Splitting**:
   - Splits the dataset into training and testing sets.

4. **Data Formatting**:
   - Formats the data into the required input-output structure for the neural network.

## Usage

1. **Train the Neural Network**:
   - Run the `titanicdataset.ipynb` notebook to preprocess the data and train the neural network.

2. **Generate Predictions**:
   - Use the trained network to generate predictions for the test dataset.
   - Save the predictions in `submission.csv` for submission.

## Example

```python
from lou import NeuralNetwork

# Define the network structure
network = NeuralNetwork([7, 10, 1])

# Train the network
network.train(train_data, alpha=0.01, tries=1000)

# Evaluate the network
print(f"Mean cost: {network.meancost(test_data)}")
```

## Requirements

- Python 3.8+
- NumPy
- Jupyter Notebook
