import tensorflow as tf
import numpy as np
from util import *

from config import Config

import os
from dataset import get_toy_data, one_hot_encode

from graph import build_graph
from model import StateTuple


input_size = 20

np.random.seed(43)
tf.set_random_seed(43)


x_v, target_v = get_toy_data(input_size, 2000)
y_v = one_hot_encode(target_v)
output_size = y_v.shape[1]
batch_size = x_v.shape[0]



np.random.seed(43)
tf.set_random_seed(43)


layer_size = 50 

config = Config(
	tau = 5.0,
	tau_gamma = 1.0,
	tau_m = 1000.0,
	adapt_gain = 1000.0,
	h = 0.2,
	gamma = tf.placeholder(tf.float32, shape=(), name="gamma"),
	num_iter = 30,
	lrate = 1.0,
)

gamma_v = 0.01



gt, gdt = build_graph(input_size, (layer_size, output_size), config)

############

sess = tf.Session()
sess.run(tf.global_variables_initializer())



init_state_fn, init_long_state_fn = gt.net.get_zero_states_fn(batch_size)


train_num_batches, test_num_batches = 1, 1


def run(e, gamma_v, x_v, y_v, states_v, a_m_v, batch_size, is_train=True):
	feeds = {
		gt.x: x_v,
		gt.y: y_v,
		gt.is_train: is_train,	
		config.gamma: gamma_v
	}
	for l, dst_s in zip(gt.net, states_v):
		feeds[l.state] = dst_s

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
	a_m_v = init_long_state_fn()


	for e in xrange(500):
		acc_err = np.zeros((len(gt.net),))
		
		train_err_rate, test_err_rate = 0.0, 0.0
		
		for bi in xrange(train_num_batches):
			init_states_v = init_state_fn()
	
			sess_res = run(e, gamma_v, x_v, y_v, init_states_v, a_m_v, batch_size)			
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
		
		if e % 5 == 0:
			for bi in xrange(test_num_batches):
	 			test_sess_res = run(e, 0.0, x_v, y_v, init_state_fn(), a_m_v, batch_size, is_train=False)			
				test_err_rate += test_sess_res[1]/test_num_batches

			print "Epoch {}, SE {}, train |E| = {:.4f}, test |E| = {:.4f}, gamma = {:.4f}".format(
				e, 
				", ".join(["{:.4f}".format(se_l) for se_l in se_acc]), 
				train_err_rate,
				test_err_rate,
				gamma_v
			)
			
			# break
		
		# gamma_v *= 1.0/(1.0+0.0001*100.0)


except KeyboardInterrupt:
	pass



sess_res = test_sess_res
rec_last_layer = sess_res[4][-1][-1][1]

read_d = lambda d, li, si: np.asarray([st[li][si] for st in d])

l0_s_acc = read_d(sess_res[3], 0, 1)
l1_s_acc = read_d(sess_res[3], 1, 1)

shs(l0_s_acc[-1], labels=(np.argmax(y_v,1),))

shm(rec_last_layer[:20], y_v[:20])

# shm(sess.run(tf.nn.softmax(s_acc[0][:10])), y_v[:10])

# test("final")
# shl(s_acc[:,1,0,:])