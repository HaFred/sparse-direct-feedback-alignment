
import tensorflow as tf
import numpy as np
from collections import namedtuple

def xavier_init(shape):
	init_range = 2.0 * np.sqrt(6. / np.sum(shape))
	return init_range * np.random.uniform(size=shape) - init_range/2.0

def bounded_relu(x):
	return tf.minimum(tf.nn.relu(x), 1.0)

StateTuple = namedtuple("StateTuple", ("u", "a"))
DebugTuple = namedtuple("DebugTuple", ("residuals", "reconstruction"))
FinalStateTuple = namedtuple("FinalStateTuple", ("u", "a", "a_m"))



class PredictiveCodingLayer(object):
	def __init__(
		self, 
		batch_size, 
		input_size,
		layer_size,
		feedback_size, 
		act,
		c,
	):
		self.batch_size = batch_size
		self.input_size = input_size
		self.layer_size = layer_size
		self.feedback_size = feedback_size

		self.h = c.h
		self.gamma = c.gamma
		self.adapt_gain = c.adapt_gain
		self.tau = c.tau
		self.tau_m = c.tau_m
		self.tau_gamma = c.tau_gamma

		self.act = act

		self.D_init = xavier_init((input_size, layer_size))
		self.D = tf.Variable(self.D_init, dtype=tf.float32)

		self.D_fb_init = -0.1 + 0.2*np.random.randn(feedback_size, layer_size)
		self.D_fb = tf.Variable(self.D_fb_init, dtype=tf.float32)

		self.a_m = tf.placeholder(tf.float32, shape=(layer_size, ))

		self._state = StateTuple(
			tf.placeholder(tf.float32, shape=(None, self.layer_size), name="u"),
			tf.placeholder(tf.float32, shape=(None, self.layer_size), name="a")
		)

	def __call__(self):
		raise NotImplementedError


	def feedback(self, state, top_down_signal):
		u, _ = state
		
		gain_td = tf.matmul(top_down_signal, self.D_fb)
		
		new_u = u + self.h * (-u - self.gamma * gain_td)/self.tau_gamma
		
		return StateTuple(new_u, self.act(new_u))

	@property
	def state(self):
		return self._state  

	def final_state(self, state):
		return FinalStateTuple(
			state.u, 
			state.a, 
			(self.adapt_gain*tf.reduce_mean(state.a, 0) - self.a_m)/self.tau_m
		)


class ClassificationLayerNonRec(PredictiveCodingLayer):
	def __call__(self, state, x, a_t):
		
		new_u = tf.matmul(x, self.D)
		a_hat = self.act(new_u)
		
		residuals = a_t - a_hat
		
		return StateTuple(new_u, residuals), DebugTuple(residuals, a_hat)

	def get_grads_and_vars(self, state, x):
		
		dD = -tf.matmul(tf.transpose(x), state.a)
		
		return [(dD/tf.cast(self.batch_size, tf.float32), self.D)]

class ClassificationLayer(PredictiveCodingLayer):
	def __call__(self, state, x, a_t):
		u, _ = state
		
		a_hat = self.act(tf.matmul(x, self.D))
		residuals = gain = a_t - a_hat

		new_u = u + self.h * (-u + gain)/self.tau
		
		return StateTuple(new_u, new_u), DebugTuple(residuals, a_hat)


	def get_grads_and_vars(self, state, x):
		
		dD = -tf.matmul(tf.transpose(x), state.a)
		
		return [(dD/tf.cast(self.batch_size, tf.float32), self.D)]


class ReconstructionLayerNonRec(PredictiveCodingLayer):
	def __call__(self, state, x, a_t):
		u = tf.matmul(x, self.D)
		a = self.act(u)
		x_hat = tf.matmul(a, tf.transpose(self.D))

		residuals = gain = x - x_hat

		return StateTuple(new_u, residuals), DebugTuple(residuals, x_hat)

	def get_grads_and_vars(self, state, x):
		
		r = x - tf.matmul(state.a, tf.transpose(self.D))
		
		dD = -tf.matmul(tf.transpose(r), state.a)
		
		return [(dD/tf.cast(self.batch_size, tf.float32), self.D)]



class ReconstructionLayer(PredictiveCodingLayer):
	def __call__(self, state, x, a_t):
		u, a = state

		x_hat = tf.matmul(a, tf.transpose(self.D))

		residuals = x - x_hat

		gain = tf.matmul(residuals, self.D)

		new_u = u + self.h * (-u + gain)/self.tau
		
		# y = self.act(new_u - self.a_m)
		a = self.act(new_u)
		return StateTuple(new_u, a), DebugTuple(residuals, x_hat)

	def get_grads_and_vars(self, state, x):
		
		r = x - tf.matmul(state.a, tf.transpose(self.D))
		
		dD = -tf.matmul(tf.transpose(r), state.a)
		
		return [(dD/tf.cast(self.batch_size, tf.float32), self.D)]


