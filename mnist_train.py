
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
from util import *

from config import Config

import os
from dataset import get_toy_data, one_hot_encode

from graph import build_graph
from model import StateTuple


np.random.seed(43)
tf.set_random_seed(43)

input_size = 28*28
output_size = 10

layer_size = 800

config = Config(
	tau = 5.0,
	tau_gamma = 1.0,
	tau_m = 1000.0,
	adapt_gain = 1000.0,
	h = 0.2,
	gamma = 5.0,
	num_iter = 5,
	lrate = 1e-03,
)



gt, gdt = build_graph(input_size, (layer_size, output_size), config)

############

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100


init_state_fn, init_long_state_fn = gt.net.get_zero_states_fn(batch_size)

init_states_v = init_state_fn()
a_m_v = init_long_state_fn()

mnist = input_data.read_data_sets(
	"{}/tmp/MNIST_data/".format(os.environ["HOME"]),
	one_hot=True
)

train_num_batches = mnist.train.num_examples/batch_size
test_num_batches = mnist.test.num_examples/batch_size

# train_num_batches, test_num_batches = 1, 1

def run(e, x_v, y_v, a_m_v, batch_size, is_train=True):
	init_states_v = init_state_fn()

	feeds = {
		gt.x: x_v,
		gt.y: y_v,
		gt.is_train: is_train,
		tuple(l.state for l in gt.net): init_states_v
	}

	feeds.update(
		dict([(l.a_m, aml) for l, aml in zip(gt.net, a_m_v)])
	)

	sess_res = sess.run(
		(
			gt.final_states,
			gt.error_rate,
			gdt.errors, 
			gdt.states_acc,
			gdt.debug_acc,
			gdt.grads_and_vars,
		) + (
			(gt.apply_grads_step,) if is_train else ()
			# (gt.apply_grads_step,) if e > 0 and learn else ()
		),
		feeds
	)
	return sess_res		

try:

	for e in xrange(100):
		acc_err = np.zeros((len(gt.net),))
		
		train_err_rate, test_err_rate = 0.0, 0.0

		for bi in xrange(train_num_batches):
 			x_v, y_v = mnist.train.next_batch(batch_size) 

			sess_res = run(e, x_v, y_v, a_m_v, batch_size)			
			final_state_v, batch_err_rate, se_v = sess_res[:3]
			
			init_states_v = tuple(
				StateTuple(*tuple(t.copy() for t in (s.u, s.a)))
				for s in final_state_v
			)

			a_m_v = tuple(fsv.a_m for fsv in final_state_v)

			se_acc = np.asarray(se_v)
			if np.linalg.norm(se_acc) < 1e-10:
				raise KeyboardInterrupt
			
			acc_err += se_acc/train_num_batches
			train_err_rate += batch_err_rate/train_num_batches
			# if e % 5 == 0:
			# 	test(e)
			
		for bi in xrange(test_num_batches):
 			xt_v, yt_v = mnist.test.next_batch(batch_size) 

			test_sess_res = run(e, xt_v, yt_v, a_m_v, batch_size, is_train=False)			
			test_err_rate += test_sess_res[1]/test_num_batches

		print "Epoch {}, SE {}, train |E| = {:.4f}, test |E| = {:.4f}".format(
			e, 
			", ".join(["{:.4f}".format(se_l) for se_l in se_acc]), 
			train_err_rate,
			test_err_rate
		)
		# break


except KeyboardInterrupt:
	pass


rec_last_layer = sess_res[4][-1][-1][1]

read_d = lambda d, li, si: np.asarray([st[li][si] for st in d])

l0_s_acc = read_d(sess_res[3], 0, 1)
l1_s_acc = read_d(sess_res[3], 1, 1)

shs(l0_s_acc[-1], labels=(np.argmax(y_v,1),))

shm(rec_last_layer[:20], y_v[:20])

# shm(sess.run(tf.nn.softmax(s_acc[0][:10])), y_v[:10])

# test("final")
# shl(s_acc[:,1,0,:])