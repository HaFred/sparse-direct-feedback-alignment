
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *

np.random.seed(5)

act = Relu()
act_o = Linear()

input_size = 100

x = np.random.randn(input_size)

lrule = Learning.BP


y_t = np.asarray([0.8, 0.2])

net_structure = (50, 20, 10, 2)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)


W = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.1
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = list(
    np.random.random((net_structure[-1], size))*1.0
    for li, size in enumerate(net_structure[:-1])
)


fb_factor = 1.0
tau = 5.0
num_iters = 500

step = 0.1

tau_apical = 2.0
tau_basal = 2.0

lrate = 0.02


# sp_code = False
predictive_output = False

h_h = [np.zeros((num_iters, ns)) for ns in net_structure]
e_h = [np.zeros((num_iters)) for _ in net_structure]
y_h = np.zeros((num_iters))


h = [np.zeros(ns) for ns in net_structure]
e = [np.zeros(ns) for ns in (input_size, ) + net_structure[:-1]]
r = [np.zeros(ns) for ns in net_structure]

print tuple(ee.shape for ee in e)

for i in xrange(num_iters):
    if i > 250:
        fb_factor = 0.0
    
    for li in xrange(len(net_structure)-1):
        input_to_layer = x if li == 0 else r[li-1]

        e[li] = input_to_layer - np.dot(r[li], W[li].T)
        h[li] += step * (np.dot(e[li], W[li]) - fb_factor * e[li+1])/tau

        # h[li] += step * (np.dot(e[li], W[li]) - fb_factor * np.dot(r[-1]-y_t, B[li]))/tau
        r[li] = act(h[li])
        
    
    h[-1] = np.dot(r[-2], W[-1])
    r[-1] = act_o(h[-1])
    e[-1] = np.dot(r[-1] - y_t, W[-1].T)
    

    # for li in xrange(len(net_structure)-1):
    #     e[li+1] = np.dot(- (y_t - r[-1]), B[li])

    # print tuple(ee.shape for ee in e)
    # raise Exception
    error = tuple(np.linalg.norm(ee) for ee in e)

#     # dW0 = 
#     dW0 = -np.outer(in0, r0 * act.deriv(top_down0))

     # dW = tuple(-np.outer(ee, np.dot(e, W[1].T) ) for ee, li in enumerate(e))
    dW = []
    for ee, rr in zip(e, r):
        dW.append(np.outer(ee, rr))
    
    dW[-1] = np.outer(r[-2], y_t - r[-1])
    
      # dW0 = -np.outer(in0, r0 * act.deriv(top_down0))
#     dW1 = -np.outer(r0, e)
    for li in xrange(len(net_structure)):
        W[li] += lrate * dW[li]

    

    # for li, dWl in enumerate(dW):
    #     W[li] += lrate*dWl

#     W[0] -= lrate * dW0
#     W[1] -= lrate * dW1

    print "i {}, error {}".format(i, ", ".join(["{:.4f}".format(ee) for ee in error]))

    for h_hl, hl in zip(h_h, h):
        h_hl[i] = hl.copy()

    for e_hl, error_l in zip(e_h, error):
        e_hl[i] = error_l

#     h0_h[i] = h0.copy()
#     e0_h[i] = error[0]
#     h1_h[i] = h1.copy()
#     e1_h[i] = error[1]

shl(*e_h)

# shl(h_h[-1][:,0], np.asarray([y_t[0]]*num_iters),np.asarray([3.0]*num_iters), show=False)
# shl(h_h[-1][:,1], np.asarray([y_t[1]]*num_iters),np.asarray([3.0]*num_iters))
