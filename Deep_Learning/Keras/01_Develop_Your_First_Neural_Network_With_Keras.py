import numpy as np
import numpy as pd

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt("D:/01_Download/data/pima-indians-diabetes.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

print(X[0,:])
print(Y[0])
