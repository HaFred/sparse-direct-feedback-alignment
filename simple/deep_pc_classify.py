
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from gradflow.gradflow_util import *
from dataset import get_toy_data, one_hot_encode

# np.random.seed(5)

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
net_structure = (100, 20, output_size)



W = list(
    np.random.randn((net_structure[li-1] if li > 0 else input_size, size))*0.1
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

step = 0.05

lrate = 0.00005


# sp_code = False
predictive_output = False

# h_h = [np.zeros((num_iters, ns)) for ns in net_structure]

# y_h = np.zeros((num_iters))



def run(x, y, t, num_iters, fb_factor, learn, print_every):
    e_h = [np.zeros((num_iters)) for _ in net_structure]

    batch_size = x.shape[0]

    h = [np.zeros((batch_size, ns)) for ns in net_structure]
    e = [np.zeros((batch_size, ns)) for ns in (input_size, ) + net_structure[:-1]]
    r = [np.zeros((batch_size, ns)) for ns in net_structure]

    for i in xrange(num_iters):
        for li in xrange(len(net_structure)-1):
            input_to_layer = x if li == 0 else r[li-1]

            e[li] = input_to_layer - np.dot(r[li], W[li].T)
            h[li] += step * (np.dot(e[li], W[li]) - fb_factor * e[li+1])/tau

            # h[li] += step * (np.dot(e[li], W[li]) - fb_factor * np.dot(r[-1]-y, B[li]))/tau
            r[li] = act(h[li])
            
        
        h[-1] = np.dot(r[-2], W[-1])
        r[-1] = act_o(h[-1])
        e[-1] = np.dot(r[-1] - y, W[-1].T)
        

        error = tuple(np.linalg.norm(ee) for ee in e)
        
        classification_error_rate = np.mean(np.not_equal(np.argmax(r[-1], axis=1), t))
        if learn:
            dW = []
            for ee, rr in zip(e, r):
                dW.append(np.dot(ee.T, rr))
            
            dW[-1] = np.dot(r[-2].T, y - r[-1])
            
            for li in xrange(len(net_structure)):
                W[li] += lrate * dW[li]

        if (i+1) % print_every == 0:
            print "i {}, {:.4f}, error {}".format(i, classification_error_rate, ", ".join(["{:.4f}".format(ee) for ee in error]))


        for e_hl, error_l in zip(e_h, error):
            e_hl[i] = error_l


    return e_hl


for _ in xrange(10):
    print "Train"
    e_hl = run(x_v, y_v, target_v, 1000, 0.1, learn=True, print_every=1000)
    print "Test"
    e_hl = run(x_v_test, y_v_test, target_v_test, 100, 0.0, learn=False, print_every=100)


# for i in xrange(500):    
#     for li in xrange(len(net_structure)-1):
#         input_to_layer = x_v_test if li == 0 else r[li-1]

#         e[li] = input_to_layer - np.dot(r[li], W[li].T)
#         h[li] += step * np.dot(e[li], W[li])/tau

#         # h[li] += step * (np.dot(e[li], W[li]) - fb_factor * np.dot(r[-1]-y_v, B[li]))/tau
#         r[li] = act(h[li])
        
    
#     h[-1] = np.dot(r[-2], W[-1])
#     r[-1] = act_o(h[-1])
#     e[-1] = np.dot(r[-1] - y_v_test, W[-1].T)


#     error = tuple(np.linalg.norm(ee) for ee in e)

#     classification_error_rate = np.mean(np.not_equal(np.argmax(r[-1], axis=1), target_v_test))


#     print "i test {}, {:.4f}, error {}".format(i, classification_error_rate, ", ".join(["{:.4f}".format(ee) for ee in error]))





##############################









