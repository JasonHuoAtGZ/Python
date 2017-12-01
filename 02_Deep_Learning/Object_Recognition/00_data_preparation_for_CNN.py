# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# data structure exploration
print(type(X_train))
print(X_train.shape)

print(type(y_train))
print(y_train.shape)



y_train2 = np_utils.to_categorical(y_train)
print(type(y_train2))
print(y_train2.shape)

print(X_train[0,:,:,:])
print(y_train[0,:])
print(y_train2[0,:])