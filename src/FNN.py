import math
import numpy as np
import scipy
import scipy.special

def ReLU(x):
    return max(0, x)

def sigmoidFunction(x):
    return scipy.special.expit(x)

class FNN():
    def __init__(self, Wh, Wo):        
        self.Wh = Wh
        self.Wo = Wo

    def forward(self, X):
        '''
        Dado as entradas do jogo e os pesos, calcula a saída da rede.
        '''
        hidden_layer = np.matmul(X, self.Wh) # calcula saída da camada intermediária
        hidden_layer = [ReLU(x) for x in hidden_layer] # aplica função ReLu
        hidden_layer.append(1) #bias
        output = np.matmul(hidden_layer, self.Wo)[0]    # calcula saída da camada de saída
        return sigmoidFunction(output) # aplica função sigmoid e retorna
        