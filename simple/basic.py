
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *

act = Sigmoid()
act_o = Linear()

lrate = 1.0
x = 1.0
T = 1
lrule = Learning.BP

epochs = 1

y_t = 0.3


W0 = 1.0
W1 = 1.0
B0 = 1.0
fb_factor = 1.0
tau = 3.0
num_iters = 100
h = 0.5

hh = np.zeros(num_iters)
eh = np.zeros(num_iters)
yh = np.zeros(num_iters)

for epoch in xrange(epochs):
    a0 = x * W0
    h0 = act(a0)
    for i in xrange(num_iters):
    
        
        # h0 += h*(-h0 + act(a0))/tau
        # h0 += h * act(a0)

        a1 = h0 * W1 
        y = act_o(a1)
        
        # e += h*(-e + (y_t - y))/tau
        e = y_t - y

        h0 += h*fb_factor*e

        hh[i] = h0
        eh[i] = np.linalg.norm(e)
        yh[i] = y

    error = (y - y_t) ** 2.0


    dW0 = x * W1 * e * act.deriv(a0)
    dW1 = h0 * e
    
    W0 += lrate * dW0
    W1 += lrate * dW1

    print "Epoch {}, error {}".format(epoch, error)

shl(yh, np.asarray([y_t]*num_iters),np.asarray([3.0]*num_iters))