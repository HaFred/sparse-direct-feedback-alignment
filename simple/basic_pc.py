
import sys
sys.path.insert(0, "/home/alexeyche/prog/sparse-direct-feedback-alignment")
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *

np.random.seed(1)

act = Linear()
act_o = Linear()

input_size = 2
x = np.ones((input_size,))

lrule = Learning.BP


y_t = np.asarray([0.2])

net_structure = (1,1)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)


W = list(
    np.random.randn(*(net_structure[li-1] if li > 0 else input_size, size))*1.0
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = np.random.random((net_structure[-1], net_structure[-2]))*1.0

fb_factor = 0.0
tau = 10.0
num_iters = 100
h = 0.02

tau_apical = 2.0
tau_basal = 2.0

lrate = 0.0


# sp_code = False
predictive_output = False

h0_h = np.zeros((num_iters, net_structure[0]))
e0_h = np.zeros((num_iters))
h1_h = np.zeros((num_iters, net_structure[1]))
e1_h = np.zeros((num_iters))
y_h = np.zeros((num_iters))
x_hat_h = np.zeros((num_iters, input_size))
in0_h = np.zeros((num_iters, input_size))

h0 = np.zeros(net_structure[0])
r0 = np.zeros(net_structure[0])
h1 = np.zeros(net_structure[1])
r1 = np.zeros(net_structure[1])
e = np.zeros((net_structure[-1]))

for i in xrange(num_iters):
    in0 = x - np.dot(r0, W[0].T)
    top_down0 = e
    
    h0 += h * (np.dot(in0, W[0]) + fb_factor * np.dot(top_down0, W[1].T))
    
    r0 = act(h0)


    e = y_t - r1
    
    if predictive_output:
        # in1 = np.dot(e, W[1].T)
        # h1 += h * np.dot(in1, W[1])

        in1 = y_t - (r1 - np.dot(r0, W[1]))
        h1 += h * in1
    else:
        h1 = np.dot(r0, W[1])
    
    r1 = act(h1)

    error = np.asarray((
        np.sum(in0 ** 2.0),
        np.sum(e ** 2.0),
    ))


    # dW0 = -np.outer(in0, np.dot(e, W[1].T) )
    
    dW0 = -np.outer(in0, r0 * act.deriv(top_down0))
    dW1 = -np.outer(r0, e)
    
    W[0] -= lrate * dW0
    W[1] -= lrate * dW1

    print "i {}, error {}".format(i, ", ".join(["{:.4f}".format(ee) for ee in error]))

    h0_h[i] = h0.copy()
    e0_h[i] = error[0]
    h1_h[i] = h1.copy()
    e1_h[i] = error[1]
    x_hat_h[i] = np.dot(r0, W[0].T)
    in0_h[i] = in0.copy()

# shl(h1_h, np.asarray([y_t]*num_iters),np.asarray([3.0]*num_iters))
shl(e0_h)

