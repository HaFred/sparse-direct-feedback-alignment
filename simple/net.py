
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *

np.random.seed(4)

act = Linear()
act_o = Linear()

input_size = 10
x = np.ones((input_size,))

lrule = Learning.BP


y_t = 0.3

net_structure = (5,1)


W = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.2
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = np.random.random((net_structure[-1], net_structure[-2]))*0.2

fb_factor = 1.0
tau = 3.0
num_iters = 100
h = 1.0

epochs = 100
lrate = 0.05


sp_code = False
feedback = False

if not sp_code and not feedback:
    num_iters = 1

hh = np.zeros((epochs * num_iters, net_structure[0]))
eh = np.zeros((epochs * num_iters))
yh = np.zeros((epochs * num_iters))
gh = np.zeros((epochs * num_iters, net_structure[0]))
rh = np.zeros((epochs * num_iters))

for epoch in xrange(epochs):
    a0 = np.dot(x, W[0])
    
    h0 = act(a0)

    for i in xrange(num_iters):
        # affects: h0, e, y, a1
        if sp_code:
            h0 += h * act(np.dot(x - np.dot(h0, W[0].T), W[0]))
        
        
        a1 = np.dot(h0, W[1])
        y = act_o(a1)
        
        e = y_t - y
        
        if feedback:
            h0 += h*fb_factor*np.dot(e, B)

        
        hh[epoch * num_iters + i] = h0.copy()
        eh[epoch * num_iters + i] = np.linalg.norm(e)
        yh[epoch * num_iters + i] = y.copy()
        rh[epoch * num_iters + i] = np.linalg.norm(act(x - np.dot(h0, W[0].T)))

        if not sp_code and not feedback:
            break
    
    error = (act_o(np.dot(act(a0), W[1])) - y_t) ** 2.0
    if error > 100.0:
        raise Exception
    
    # dW0 = np.outer(x, h0 * act.deriv(a0))
    # dW0 = np.outer(x, np.dot(e, B) * act.deriv(a0))
    dW0 = np.outer(x, np.dot(np.dot(h0, W[0].T) - x, W[0]))
    
    dW1 = np.outer(h0, e)
    
    W[0] += lrate * dW0
    # W[1] += lrate * dW1

    print "Epoch {}, error {}".format(epoch, error)

shl(yh, np.asarray([y_t]*num_iters),np.asarray([3.0]*num_iters))