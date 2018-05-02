from Layer import Layer
import numpy as np
import random
import sys

import matplotlib.pyplot as plt

def del_func(del_fwd, out_vect, weights_fwd):
    return_vect = np.zeros(out_vect.shape)
    for i in range(out_vect.size):
        del_sum = 0
        for j in range(weights_fwd.shape[0]):
            del_sum += del_fwd[j] * weights_fwd[j, i]
        return_vect[i] = out_vect[i] * (1 - out_vect[i]) * del_sum
    return return_vect


def del_func_top(out_desired, out_vect):
    if out_desired.size != out_vect.size:
        sys.exit("In del_func_top: Vectors must be the same size!")
    if out_desired.size == 1:
        return np.array((out_desired - out_vect) * out_vect * (1 - out_vect))
    return_vect = np.zeros(out_desired.shape)
    for i in range(out_desired.size):
        return_vect[i] = (out_desired[i] - out_vect[i]) * out_vect[i] * (1 - out_vect[i])
    return return_vect


def err_eval(vec1, vec2):
    if vec1.size != vec2.size:
        sys.exit("In err_eval: Vectors must be the same size!")
    if vec1.size == 1:
        return 0.5 * (vec1 - vec2)**2
    ret_val = 0
    for i in range(vec1.size):
        ret_val += 0.5 * (vec1[i] - vec2[i])**2
    return ret_val


n = .25
rounds = 30000
random.seed(1)
init_vect = np.zeros((1, 1))
lay1 = Layer(init_vect, 3)
lay2 = Layer(lay1.out_vect, 20)
lay3 = Layer(lay2.out_vect, 1)
printmod = rounds/1000
x = list(range(0, 1000))
y = [0] * 1000

for i in range(rounds):

    #init_vect = np.array([[random.randint(0,1)], [random.randint(0,1)]])
    #out_desired_bool = bool(init_vect[0]) == bool(init_vect[1])
    #out_desired = np.array([[int(out_desired_bool)]])
    #out_desired = init_vect
    init_vect = np.array(random.uniform(0.2, 0.9))
    out_desired = np.array(float(init_vect) ** 2)
    lay1.set_in(init_vect)
    lay2.set_in(lay1.out_vect)
    lay3.set_in(lay2.out_vect)

    out_val = lay3.out_vect

    err = float(err_eval(out_desired, out_val))
    #out_desired = bool(init_vect[0]) != bool(init_vect[1])
    #out_desired = int(out_desired)

    if i % printmod == 0:
        y[int(i/printmod)] = err
        print("%s pct done" % int(100*i/rounds))
        print(err)

    del_3 = del_func_top(out_desired, out_val)
    del_2 = del_func(del_3, lay2.out_vect, lay3.weights)
    del_1 = del_func(del_2, lay1.out_vect, lay2.weights)

    weight_grad_1 = n * del_1.dot(lay1.in_vect.transpose())
    weight_grad_2 = n * del_2.dot(lay2.in_vect.transpose())
    weight_grad_3 = n * del_3.dot(lay3.in_vect.transpose())

    bias_grad_1 = n * del_1
    bias_grad_2 = n * del_2
    bias_grad_3 = n * del_3
    lay1.update_grad(weight_grad_1, bias_grad_1)
    lay2.update_grad(weight_grad_2, bias_grad_2)
    lay3.update_grad(weight_grad_3, bias_grad_3)


def testnn(in_vec):
    lay1.set_in(in_vec)
    lay2.set_in(lay1.out_vect)
    lay3.set_in(lay2.out_vect)
    return lay3.out_vect

#plt.plot(x,y)
#plt.show()

