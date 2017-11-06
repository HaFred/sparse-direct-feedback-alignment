
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *
from dataset import get_toy_data, one_hot_encode

np.random.seed(5)

input_size = 20
x_v, target_v = get_toy_data(input_size, 2000, seed = 5)


test_prop = x_v.shape[0]/5

x_v_test = x_v[:test_prop]
target_v_test = target_v[:test_prop]

x_v = x_v[test_prop:]
target_v = target_v[test_prop:]

####

y_v = one_hot_encode(target_v)
y_v_test = one_hot_encode(target_v_test)


output_size = y_v.shape[1]


act = Relu()
act_o = Linear()

batch_size = x_v.shape[0]
test_batch_size = x_v_test.shape[0]
net_structure = (100, output_size)



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
num_iters = 5000

step = 0.1

lrate = 0.001


# sp_code = False
predictive_output = False

# h_h = [np.zeros((num_iters, ns)) for ns in net_structure]
e_h = [np.zeros((num_iters)) for _ in net_structure]
# y_h = np.zeros((num_iters))


h = [np.zeros((batch_size, ns)) for ns in net_structure]
e = [np.zeros((batch_size, ns)) for ns in (input_size, ) + net_structure[:-1]]
r = [np.zeros((batch_size, ns)) for ns in net_structure]


for i in xrange(num_iters):  
    # if i > 500:
    #     fb_factor = 0.0  
    for li in xrange(len(net_structure)-1):
        input_to_layer = x_v if li == 0 else r[li-1]

        e[li] = input_to_layer - np.dot(r[li], W[li].T)
        h[li] += step * (np.dot(e[li], W[li]) - fb_factor * e[li+1])/tau

        # h[li] += step * (np.dot(e[li], W[li]) - fb_factor * np.dot(r[-1]-y_v, B[li]))/tau
        r[li] = act(h[li])
        
    
    h[-1] = np.dot(r[-2], W[-1])
    r[-1] = act_o(h[-1])
    e[-1] = np.dot(r[-1] - y_v, W[-1].T)
    

    # for li in xrange(len(net_structure)-1):
    #     e[li+1] = np.dot(- (y_v - r[-1]), B[li])

    # print tuple(ee.shape for ee in e)
    # raise Exception
    error = tuple(np.linalg.norm(ee) for ee in e)
    
    classification_error_rate = np.mean(np.not_equal(np.argmax(r[-1], axis=1), target_v))

#     # dW0 = 
#     dW0 = -np.outer(in0, r0 * act.deriv(top_down0))

     # dW = tuple(-np.outer(ee, np.dot(e, W[1].T) ) for ee, li in enumerate(e))
    dW = []
    for ee, rr in zip(e, r):
        dW.append(np.dot(ee.T, rr))
    
    dW[-1] = np.dot(r[-2].T, y_v - r[-1])
    
      # dW0 = -np.outer(in0, r0 * act.deriv(top_down0))
#     dW1 = -np.outer(r0, e)
    for li in xrange(len(net_structure)):
        W[li] += lrate * dW[li]

    

    # for li, dWl in enumerate(dW):
    #     W[li] += lrate*dWl

#     W[0] -= lrate * dW0
#     W[1] -= lrate * dW1

    print "i {}, {:.4f}, error {}".format(i, classification_error_rate, ", ".join(["{:.4f}".format(ee) for ee in error]))

    # for h_hl, hl in zip(h_h, h):
    #     h_hl[i] = hl.copy()

    for e_hl, error_l in zip(e_h, error):
        e_hl[i] = error_l

#     h0_h[i] = h0.copy()
#     e0_h[i] = error[0]
#     h1_h[i] = h1.copy()
#     e1_h[i] = error[1]

# shl(*e_h)

# shl(h_h[-1][:,0], np.asarray([y_v[0]]*num_iters),np.asarray([3.0]*num_iters), show=False)
# shl(h_h[-1][:,1], np.asarray([y_v[1]]*num_iters),np.asarray([3.0]*num_iters))


h = [np.zeros((test_batch_size, ns)) for ns in net_structure]
e = [np.zeros((test_batch_size, ns)) for ns in (input_size, ) + net_structure[:-1]]
r = [np.zeros((test_batch_size, ns)) for ns in net_structure]


for i in xrange(500):    
    for li in xrange(len(net_structure)-1):
        input_to_layer = x_v_test if li == 0 else r[li-1]

        e[li] = input_to_layer - np.dot(r[li], W[li].T)
        h[li] += step * np.dot(e[li], W[li])/tau

        # h[li] += step * (np.dot(e[li], W[li]) - fb_factor * np.dot(r[-1]-y_v, B[li]))/tau
        r[li] = act(h[li])
        
    
    h[-1] = np.dot(r[-2], W[-1])
    r[-1] = act_o(h[-1])
    e[-1] = np.dot(r[-1] - y_v_test, W[-1].T)


    error = tuple(np.linalg.norm(ee) for ee in e)

    classification_error_rate = np.mean(np.not_equal(np.argmax(r[-1], axis=1), target_v_test))


    print "i test {}, {:.4f}, error {}".format(i, classification_error_rate, ", ".join(["{:.4f}".format(ee) for ee in error]))





##############################









