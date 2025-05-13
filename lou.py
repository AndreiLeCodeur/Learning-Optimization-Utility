# Library used to generate own simple neural network
# Activation function : sigmoid
# Cost function :  quadratic mean

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:

    def __init__(self, struct):
        '''
        struct is a list of integers symbolizing the size of each layer 
        '''
        self.Biases, self.Weights = [], []
        for i in range(1, len(struct)):
            self.Weights.append(np.random.rand(struct[i],struct[i-1]))
            self.Biases.append(np.random.rand(struct[i],1))
        self.length = len(struct)

    def frontprop(self, A0):
        '''
        A0 : input, numpy array of dimension (struct[0],1)
        output : each layer and the last ... in a list (contains the input)
        '''
        output = [A0.reshape(-1,1)]
        for i in range(self.length-1):
            output.append(sigmoid(self.Weights[i] @ output[i] + self.Biases[i]))
        return output
    
    def train(self, data, alpha=0.01, tries=10000):
        for trial in range(tries):
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
        totalcost = 0
        for line in data:
            totalcost += np.sum(((self.frontprop(line[0])[-1]-line[1])**2))/2
        return totalcost/len(data)