import math
import numpy as np

def ReLU(x):
    return max(0, x)

def sigmoidFunction(x):
    if x > 500:
        return 1
    elif x < - 500: 
        return 0
    return 1/(1+ math.exp(x*-1))

class FNN():
    def __init__(self, Wh, Wo):        
        self.Wh = Wh
        self.Wo = Wo

    def forward(self, X):
        hidden_layer = np.matmul(X, self.Wh)
        hidden_layer = [ReLU(x) for x in hidden_layer]
        hidden_layer.append(1) #bias
        output = np.matmul(hidden_layer, self.Wo)[0]
        return sigmoidFunction(output)
        