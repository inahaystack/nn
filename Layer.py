import numpy as np
import math

import sys

np.random.seed(1)
class Layer(object):
    def __init__(self, in_vect, out_dim):
        self.in_vect = in_vect
        self.in_dim = self.in_vect.size
        self.out_dim = out_dim
        self.out_vect = np.zeros((out_dim, 1))  #initialize output vector to 0's
        self.weights = np.random.rand(out_dim, self.in_dim)
        self.bias = np.random.rand(out_dim, 1)
        self.eval()

    def sig(self, num):
        return 1/(1+math.exp(-num))

    def eval(self):
        sum_vect = self.weights.dot(self.in_vect) + self.bias
        sigfunc = np.vectorize(self.sig)
        self.out_vect = sigfunc(sum_vect)
        if self.out_vect.shape[1] != 1:
            sys.exit("Output vector is not a vector!")

    def set_in(self, new_in_vect):
        self.in_vect = new_in_vect
        self.eval()

    def update_grad(self, weight_grad, bias_grad):
        self.weights = self.weights + weight_grad
        self.bias = self.bias + bias_grad

    def out_vect(self):
        return self.out_vect
