
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *
import os

from pylab import rcParams
rcParams['figure.figsize'] = 25, 20

np.random.seed(4)

act = Linear()
act_o = Linear()

input_size = 1
x = np.ones((input_size,))

lrule = Learning.BP


y_t = np.asarray([0.8])

net_structure = (1,1)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)


B = np.random.random((net_structure[-1], net_structure[-2]))*1.0

fb_factor = 1.0
tau = 5.0
h = 0.1

tau_apical = 2.0
tau_basal = 2.0


slice_size = 25
num_iters = 200
predictive_output = False
lrate = 0.1

classic_neuron = False

h0_h = np.zeros((num_iters, net_structure[0]))
e0_h = np.zeros((num_iters))
h1_h = np.zeros((num_iters, net_structure[1]))
e1_h = np.zeros((num_iters))
y_h = np.zeros((num_iters))



W0a = np.linspace(-1.5, 1.5, slice_size)
W1a = np.linspace(-1.5, 1.5, slice_size)

W0res = np.zeros((num_iters, slice_size, slice_size))
W1res = np.zeros((num_iters, slice_size, slice_size))
dW0res = np.zeros((num_iters, slice_size, slice_size))
dW1res = np.zeros((num_iters, slice_size, slice_size))
r0res = np.zeros((num_iters, slice_size, slice_size))
error_res = np.zeros((num_iters, slice_size, slice_size))

h0_h = np.zeros((num_iters, slice_size, slice_size, net_structure[0]))
e0_h = np.zeros((num_iters, slice_size, slice_size))
h1_h = np.zeros((num_iters, slice_size, slice_size, net_structure[1]))
e1_h = np.zeros((num_iters, slice_size, slice_size))

for ri, W0 in enumerate(W0a):
    for ci, W1 in enumerate(W1a):

        W = list([
            np.asarray([W0]).reshape(input_size, 1),
            np.asarray([W1]).reshape(1, 1)
        ])

        if classic_neuron:
            for i in xrange(num_iters):
                h0 = np.dot(x, W[0])
                r0 = act(h0)

                h1 = np.dot(r0, W[1])
                r1 = act(h1)

                e = y_t - r1
                error = np.linalg.norm(e)

                dW0 = -np.outer(x, np.dot(W1, e) * act.deriv(h0))
                dW1 = -np.outer(r0, e)

                W0res[i, ri, ci] = W0
                W1res[i, ri, ci] = W1
                dW0res[i, ri, ci] = -dW0
                dW1res[i, ri, ci] = -dW1
                
                error_res[i, ri, ci] = error
                
                W[0] -= lrate * dW0
                W[1] -= lrate * dW1

        else:

            h0 = np.zeros(net_structure[0])
            r0 = np.zeros(net_structure[0])
            h1 = np.zeros(net_structure[1])
            r1 = np.zeros(net_structure[1])

            e = np.zeros((net_structure[-1]))
            

            for i in xrange(num_iters):
                in0 = x - np.dot(r0, W[0].T)
                top_down0 = e
                
                h0 += h * (np.dot(in0, W[0]) - fb_factor * np.dot(top_down0, W[1].T))
                
                r0 = act(h0)


                e = y_t - r1
                
                if predictive_output:
                    in1 = np.dot(e, W[1].T)
                    h1 += h * np.dot(in1, W[1])
                else:
                    h1 = np.dot(r0, W[1])
                
                r1 = act(h1)

                error = np.asarray((
                    np.sum(in0 ** 2.0),
                    np.sum((y_t - r1) ** 2.0),
                ))


                # dW0 = -np.outer(in0, np.dot(e, W[0].T) )
                dW0 = -np.outer(in0, h0 * act.deriv(top_down0))
                dW1 = -np.outer(r0, e)
            
                h0_h[i, ri, ci] = h0.copy()
                e0_h[i, ri, ci] = error[0]
                h1_h[i, ri, ci] = h1.copy()
                e1_h[i, ri, ci] = error[1]


                W0res[i, ri, ci] = W0
                W1res[i, ri, ci] = W1
                dW0res[i, ri, ci] = -dW0
                dW1res[i, ri, ci] = -dW1
                
                error_res[i, ri, ci] = error[-1]
        
                W[0] -= lrate * dW0
                W[1] -= lrate * dW1

# dW0res = np.clip(dW0res, -0.5, 0.5)
# dW1res = np.clip(dW1res, -0.5, 0.5)

def descr():
    s = ""
    if classic_neuron:
        s += "_class"
    return s


for i in xrange(num_iters):

    plt.quiver(
        W0res[i], W1res[i], dW0res[i], dW1res[i],
        error_res[i],
        cmap=cm.seismic,
        headlength=7, headwidth=5.0)

    plt.colorbar()
    plt.savefig("{}/tmp/pics/grad_flow{}_{}.png".format(os.environ["HOME"], descr(), i))
    plt.clf()
