import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import Sigmoid, Linear, norm, Learning

act = Linear()

lrate = 1e-01
x = 1.0
T = 1
lrule = Learning.BP

epochs = 1

y_t = 0.5


W0 = 1.0
W1 = 1.0
B0 = 1.0
slice_size = 25
fb_factor = 0.0
keep_factor = 1.0 - fb_factor

W0a = np.linspace(-1, 1.0, slice_size)
W1a = np.linspace(-1, 1.0, slice_size)

W0res = np.zeros((slice_size, slice_size))
W1res = np.zeros((slice_size, slice_size))
dW0res = np.zeros((slice_size, slice_size))
dW1res = np.zeros((slice_size, slice_size))
error_res = np.zeros((slice_size, slice_size))

for ri, W0 in enumerate(W0a):
    for ci, W1 in enumerate(W1a):

        # W0, W1 = 0.5, 0.5
        for epoch in xrange(epochs):

            a0 = x * W0; h0 = act(a0)
            a1 = h0 * W1; y = act(a1)

            e = y_t - y

            h0 = h0 * keep_factor + e * fb_factor

            error = (y - y_t) ** 2.0

            if lrule == Learning.BP:
                
                dW0 = x * W1 * e * act.deriv(a0)
                dW1 = h0 * e

            elif lrule == Learning.FA:
                
                dW0 = x * B0 * e * act.deriv(a0)
                dW1 = h0 * e

            elif lrule == Learning.HEBB:
                
                dW0 = x * h0
                dW1 = h0 * e
                                
                
            
            W0res[ri, ci] = W0
            W1res[ri, ci] = W1
            dW0res[ri, ci] = dW0
            dW1res[ri, ci] = dW1
            
            error_res[ri, ci] = error

            # W0 -= lrate * dW0
            # W1 -= lrate * dW1

dW0res = np.clip(dW0res, -1.0, 1.0)
dW1res = np.clip(dW1res, -1.0, 1.0)

plot = plt.figure()
plt.quiver(
    W0res, W1res, dW0res, dW1res,
    error_res,
    cmap=cm.seismic,     # colour map
    headlength=7, headwidth=5.0)        # length of the arrows

plt.colorbar()                      # add colour bar on the right
plt.show(plot)