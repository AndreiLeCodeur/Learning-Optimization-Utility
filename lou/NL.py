# Library used to generate own simple neural network
# Activation function : sigmoid
# Cost function :  quadratic mean

import numpy as np
import os

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:
    """
    NeuralNetwork: A class for implementing a simple feedforward neural network with backpropagation.
    Attributes:
        Biases (list): A list of numpy arrays representing the biases for each layer.
        Weights (list): A list of numpy arrays representing the weights for each layer.
        length (int): The number of layers in the neural network.
    Methods:
        __init__(struct):
            Initializes the neural network with random weights and biases based on the given structure.
            Args:
                struct (list): A list of integers representing the number of neurons in each layer.
        frontprop(A0):
            Performs forward propagation through the network.
            Args:
                A0 (numpy.ndarray): The input to the network, a numpy array of shape (struct[0], 1).
            Returns:
                list: A list of numpy arrays representing the activations of each layer, including the input.
        train(data, alpha=0.01, tries=10000):
            Trains the neural network using backpropagation and gradient descent.
            Args:
                data (list): A list of tuples, where each tuple contains an input (numpy array) and its corresponding target output (numpy array).
                alpha (float, optional): The learning rate for gradient descent. Default is 0.01.
                tries (int, optional): The number of training iterations. Default is 10000.
        meancost(data):
            Computes the mean cost (mean squared error) of the network on the given dataset.
            Args:
                data (list): A list of tuples, where each tuple contains an input (numpy array) and its corresponding target output (numpy array).
            Returns:
                float: The mean cost of the network on the dataset.
    """


    def __init__(self, struct=None, Weights=None, Biases=None):
        '''
        struct is a list of integers symbolizing the size of each layer 
        '''
        if not Weights and not Biases:
            self.Biases, self.Weights = [], []
            for i in range(1, len(struct)):
                self.Weights.append(np.random.rand(struct[i],struct[i-1]))
                self.Biases.append(np.random.rand(struct[i],1))
            self.length = len(struct)
        else:
            self.Weights = Weights
            self.Biases = Biases
            self.length = len(Weights)+1

    def frontprop(self, A0):
        '''
        A0 : input, numpy array of dimension (struct[0],1)
        output : each layer and the last ... in a list (contains the input)
        '''
        output = [A0.reshape(-1,1)]
        for i in range(self.length-1):
            output.append(sigmoid(self.Weights[i] @ output[i] + self.Biases[i]))
        return output
    
    def train(self, data, alpha=0.01, tries=10000, verbose=0):
        """
        Trains the neural network using the provided dataset through backpropagation.
        Args:
            data (list of tuples): A list of training examples, where each example is a tuple (x, y).
                - x: Input vector for the neural network.
                - y: Expected output vector corresponding to the input.
            alpha (float, optional): The learning rate for gradient descent. Defaults to 0.01.
            tries (int, optional): The number of training iterations (epochs). Defaults to 10000.
        Process:
            - Performs forward propagation to compute the output of the network.
            - Computes gradients for weights and biases using backpropagation.
            - Updates weights and biases using gradient descent.
        Notes:
            - The network assumes sigmoid activation functions for all layers.
            - Gradients are calculated layer by layer, starting from the output layer and moving backward.
        Raises:
            ValueError: If the dimensions of the input data, weights, or biases are inconsistent.
        Returns:
            None
        """
        for trial in range(tries):

            if trial % verbose == 0 and verbose != 0 :
                print(f"Loop number {trial}")

            for x, y in data:
                out = self.frontprop(x)

                # Temporary lists for gradients
                dWs = [None] * len(self.Weights)
                dBs = [None] * len(self.Biases)

                # Backpropagation
                for node in range(len(out) - 1, 0, -1):  # from output to first hidden
                    if node == len(out) - 1:
                        # Output layer
                        dZ = (out[node] - y) * out[node] * (1 - out[node])
                    else:
                        # Hidden layers
                        dZ = (self.Weights[node].T @ dZ) * out[node] * (1 - out[node])

                    dW = dZ @ out[node - 1].T
                    dB = dZ

                    dWs[node - 1] = dW
                    dBs[node - 1] = dB

                # Gradient step
                for i in range(len(self.Weights)):
                    self.Weights[i] -= alpha * dWs[i]
                    self.Biases[i] -= alpha * dBs[i]

    def meancost(self, data):
        """
        Calculate the mean cost (average loss) for a given dataset.

        This method computes the mean squared error cost for the provided dataset.
        Each data point in the dataset is processed through the network's forward
        propagation, and the squared difference between the predicted output and
        the actual target is calculated and accumulated. The final cost is averaged
        over the total number of data points.

        Args:
            data (list of tuples): A list where each element is a tuple (input, target).
                - input: The input data for the neural network (e.g., numpy array).
                - target: The expected output corresponding to the input (e.g., numpy array).

        Returns:
            float: The mean cost (average loss) for the dataset.
        """
        totalcost = 0
        for line in data:
            totalcost += np.sum(((self.frontprop(line[0])[-1]-line[1])**2))/2
        return totalcost/len(data)
    
    def export(self, path):
        if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        if not os.path.isdir(path):
            print("Please provide a directory in which your model (and only model) will be stored.")
            return False
        else:
            np.savez(f'{path}/weights.npz', *self.Weights)
            np.savez(f'{path}/biases.npz', *self.Biases)
            print("Export succesful")
            return True
        
    def FeedTokens(self, Tokenizer, sentence):
        """
        Tokenize `sentence` (a list of ints) into chunks of size input_size,
        pad the final chunk to full length, run each chunk through frontprop,
        and collect the outputs.
        """
        t_sentence = Tokenizer(sentence)
        input_size = self.Weights[0].shape[1]

        res = []
        last_chunk_index = 0

        # Process full chunks
        for i in range(0, len(t_sentence) - input_size + 1, input_size):
            chunk = t_sentence[i : i + input_size]
            # reshape to (input_size, 1)
            col = np.array(chunk).reshape(input_size, 1)
            out = self.frontprop(col)[-1].flatten().tolist()
            res.append(out)
            last_chunk_index = i + input_size

        # Process the final (possibly partial) chunk
        if last_chunk_index < len(t_sentence):
            last_chunk = t_sentence[last_chunk_index:]
            # pad with zeros up to input_size
            pad_len = input_size - len(last_chunk)
            last_chunk = last_chunk + [0] * pad_len

            col = np.array(last_chunk).reshape(input_size, 1)
            out = self.frontprop(col)[-1].flatten().tolist()
            res.append(out)
        return res

def ImportNetwork(path):
    '''
    Returns  Network with the weights and biases found in the path directory
    '''
    Weights = np.load(f'{path}/weights.npz')
    Biases = np.load(f'{path}/biases.npz')

    Weights = [Weights[f] for f in Weights]
    Biases = [Biases[f] for f in Biases]

    return NeuralNetwork(Weights=Weights, Biases=Biases)

