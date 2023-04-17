import numpy as np


class Network():
    #Structure of the neural Network
    #Bias is introduced - Number of Hidden layers can affect output
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
    #Set all weights and biases
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #2d array with dimension (input x hidden)
        self.b1 = np.zeros((1, self.hiddenSize)) #2D array with dimension (1 x hidden size)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #2d array with dimension (hidden x Output)
        self.b2 = np.zeros((1, self.outputSize)) #2D array with dimension (1 x output size)


