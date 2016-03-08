import numpy as np
class Layer:
    weight = []
    bias = []

    def __init__(self, neurons, Ninputs):
        self.weight = np.random.randn(*(neurons, Ninputs)) * 0.01
        self.bias = np.random.randn(*(1, neurons)) * 0.01

    def getWeight(self):
        return self.weight

    def setWeight(self, newWeight):
        self.weight = newWeight

    def getBias(self):
        return self.bias

    def setBias(self, newBias):
        self.bias = newBias