import numpy as np

def neuralNetwork(X, Wh, Wo):
    hidden_layer = np.matmul(X, Wh)
    output = np.matmul(hidden_layer, Wo)
    return output




X = [1,2,3,4,5,6,7]

Wh = np.ones((7,4))

h = np.matmul(X, Wh)
Wo = np.array([2,2,2,2])
res = np.matmul(h, Wo)
import math
def sigmoidFunction(x):
    return 1/(1+ math.exp(x*-1))



# np.random.seed(10)

def generatePopulation(N):
    pop = [(np.random.random((7,4)), np.random.random((4,1))) for _ in range(N)]
    return pop

Wh_pai = np.random.random((7,4))
Wh_mae = np.random.random((7,4))

print(np.random.random(1)[0])