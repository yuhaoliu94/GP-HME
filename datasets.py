import numpy as np

## DataSet class 
class DataSet():

    def __init__(self, X, Y, Name):
        self._num_examples = X.shape[0]
        self._X = X
        self._Y = Y
        self._index = 0
        self._Din = X.shape[1]
        self._Dout = Y.shape[1]
        self._name = Name

    def next_sample(self, batch_size):
        tmp = self._index
        self._index += 1
        return self._X[tmp,:], self._Y[tmp,:]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def Din(self):
        return self._Din

    @property
    def Dout(self):
        return self._Dout

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def name(self):
        return self._name


