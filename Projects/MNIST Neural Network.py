import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model


class Network():
    #Structure of the neural Network
    #Bias is introduced - Number of Hidden layers can affect output
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
    #Set all weights and biases
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #2d array with dimension (input rows x hidden columns)
        self.b1 = np.zeros((1, self.hiddenSize)) #2D array with dimension (1 row x hidden column)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #2d array with dimension (hidden rows x Output columns)
        self.b2 = np.zeros((1, self.outputSize)) #2D array with dimension (1 row x output column)

    def sigmoid(self, x):
        return 1 / 1 + (np.exp(-x))  #logistic function, is the function y=1/(1+e^(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 #neuron value before activiation
        self.a1 = self.sigmoid(self.z1) #after activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2 #neuron value before activiation
        self.a2 = self.sigmoid(self.z2) #after activation
        return self.a1, self.a2
    
    def loss(self, X, y):
        y_pred = self.forward(X) #prediction 
        loss = np.mean((y_pred - y)**2) #the loss also the cost function

    def backpropogation(self, X, y):
        y_pred = self.forward(X) #prediction
        #Outut layer
        delta2 = (y_pred - y) * self.a2 * (1-self.a2) #the amount by which each output unit needs to be adjusted in order to reduce cost funciton. Basically how much each nueron effects output
        dW2 = np.dot(self.a1.T, delta2) # the gradient of the cost funtion with respect to the weights 
        db2 = np.sum(delta2, axis=0, keepdims=True)# the gradient of the cost funtion with respect to the biases. #Keepdims=True ensures dimension of gradients are consisten with dimensions of biases
        #End of Output layer
        #Hidden Layer
        delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1-self.a1)
        dW1 = np.dot(X.T, delta1) 
        db1 = np.sum(delta1, axis=0)
        #End of Hidden Layer
        return dW1, db1, dW2, db2
    
    def train(self, X, y, learning_rate, epochs):
        for i in range(epochs):
            dW1, db1, dW2, db2 = self.backpropogation(X, y)
            #updating weights and biases. The "-= " is applying the negative gradient so that the weights and biases are updated to minimize loss
            self.W1 -= learning_rate * dW1 
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            loss = self.loss(X, y)
            print(f'Epoch: {i}, loss: {loss}')
    
# Load the MNIST dataset and preprocess it
digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the NeuralNetwork class with 64 hidden units
nn = Network(inputSize=X_train.shape[1], hiddenSize=64, outputSize=len(np.unique(y_train)))

# Train the model for 1000 epochs with a learning rate of 0.1
nn.train(X = X_train, y = y_train, epochs=1000, learning_rate=0.1)

# Make predictions on the testing set and compute the accuracy
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)