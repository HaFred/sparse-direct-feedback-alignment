
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *

act = Sigmoid()
act_o = Linear()

lrate = 0.1
input_size = 10
x = np.ones((input_size,))

lrule = Learning.BP

epochs = 1

y_t = 0.3

net_structure = (5,1)


W = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.1
    for li, size in enumerate(net_structure)
)


B = np.random.random((net_structure[-1], net_structure[-2]))*0.5

fb_factor = 1.0
tau = 3.0
num_iters = 100
h = 1.0

hh = np.zeros((num_iters, net_structure[0]))
eh = np.zeros(num_iters)
yh = np.zeros(num_iters)

for epoch in xrange(epochs):
    a0 = np.dot(x, W[0])
    h0_init = act(a0)
    h0 = h0_init.copy()
    for i in xrange(num_iters):
        a1 = np.dot(h0, W[1])
        y = act_o(a1)
        
        e = y_t - y

        h0 += h*fb_factor*np.dot(e, B)

        hh[i] = h0.copy()
        eh[i] = np.linalg.norm(e)
        yh[i] = y

    error = (y - y_t) ** 2.0


    dW0 = np.outer(x, np.dot(W[1], e) * act.deriv(a0))
    dW1 = np.outer(h0, e)
    
    W[0] += lrate * dW0
    W[1] += lrate * dW1

    print "Epoch {}, error {}".format(epoch, error)

# shl(yh, np.asarray([y_t]*num_iters),np.asarray([3.0]*num_iters))