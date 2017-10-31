
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

B = np.random.random((net_structure[-1], net_structure[-2]))*1.0

fb_factor = 1.0
tau = 5.0
num_iters = 1000
h = 0.1
tau_apical = 10.0

lrate = 0.1


# sp_code = False


hh = np.zeros((num_iters, net_structure[0]))
eh = np.zeros((num_iters))
yh = np.zeros((num_iters))
gh = np.zeros((num_iters, net_structure[0]))
rh = np.zeros((num_iters))
apich = np.zeros((num_iters, net_structure[0]))


apic0 = np.zeros(net_structure[0])

h0 = np.zeros(net_structure[0])

e = np.zeros((net_structure[-1]))
for i in xrange(num_iters):
    in0 = x - np.dot(h0, W[0].T)
    # in0 = x
    a0 = np.dot(in0, W[0])
    
    apic0 = (np.dot(e, B) - h0)
    
    h0 += h * (-h0 + act(a0) + fb_factor * apic0)/tau
    
    # h0 += h * pv
    # h0 = act(a0) + pv
    
        
    a1 = np.dot(h0, W[1])
    y = act_o(a1)
        
    e = y_t - y
    
    
    error = (y - y_t) ** 2.0
    if error > 10000.0:
        raise Exception
    
    # dW0 = np.outer(x, np.dot(e, B) * act.deriv(a0))
    
    # dW0 = np.outer(x, h0 * act.deriv(a0))
    # dW0 = 0.1*np.outer(x, np.dot(in0, W[0]))
    dW0 = -0.05*np.outer(x, np.dot(np.dot(h0, W[0].T) - x, W[0]))

    dW1 = np.outer(h0, e)
    
    W[0] += lrate * dW0
    W[1] += lrate * dW1

    print "i {}, error {}".format(i, error)

    hh[i] = h0.copy()
    eh[i] = np.linalg.norm(e)
    yh[i] = y.copy()
    rh[i] = np.linalg.norm(act(x - np.dot(h0, W[0].T)))
    apich[i] = apic0.copy()

shl(yh, np.asarray([y_t]*num_iters)) #,np.asarray([3.0]*num_iters))