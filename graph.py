
import tensorflow as tf
import numpy as np

from collections import namedtuple

from model import ClassificationLayer, ClassificationLayerNonRec, ReconstructionLayer
from model import StateTuple

GraphDebugTuple = namedtuple("GraphDebugTuple", (
	"errors",
	"states_acc",
	"debug_acc",
	"grads_and_vars",
))

GraphTuple = namedtuple("GraphTuple", (
	"x", 
	"y",
	"is_train",
	"net",
	"final_states",
	"error_rate",
	"apply_grads_step"
))

class Network(list):
	def get_zero_states_fn(self, batch_size):
		return (
			lambda: tuple(
				StateTuple(*tuple(
					np.zeros([batch_size, ] + s.get_shape().as_list()[1:])
				 	for s in l.state
				))
				for l in self
			), 
			lambda: tuple(
				np.zeros(l.a_m.get_shape().as_list()) for l in self
			)
		)




def build_graph(
	input_size,
	net_structure,
	config,
	net_neuron = ReconstructionLayer,
	out_neuron = ClassificationLayer,
):
	output_size = net_structure[-1]
	x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")
	y = tf.placeholder(tf.float32, shape=(None, output_size), name="y")
	is_train = tf.placeholder(tf.bool, shape=(), name="is_train")

	batch_size = tf.shape(x)[0]
	layers_num = len(net_structure)

	
	net = Network()

	for l_id, lsize in enumerate(net_structure[:-1]):
		input_to_layer = input_size if l_id == 0 else net_structure[l_id-1]

		with tf.variable_scope("layer{}".format(l_id)):
			net.append(
				net_neuron(
					batch_size, 
					input_to_layer, 
					lsize, 
					output_size, 
					act = tf.nn.relu,
					c = config,
				)
			)

	with tf.variable_scope("layer{}".format(len(net_structure)-1)):
		net.append(
			out_neuron(
				batch_size, 
				net_structure[-2] if len(net_structure) > 1 else input_size, 
				output_size,
				output_size,
				act = tf.nn.softmax,
				c = config,
			)
		)

	####

	states = [l.state for l in net]

	debug = [None]*layers_num
	

	states_acc = []
	debug_acc = []


	for _ in xrange(config.num_iter):
		errors = []	
			
		for l_id, l in enumerate(net):
			input_to_layer = x if l_id == 0 else states[l_id-1][-1]

			states[l_id], debug[l_id] = l(states[l_id], input_to_layer, y)
		

		# to_propagate = tf.cond(is_train, lambda: states[-1].a, lambda: tf.zeros_like(states[-1].a))
		to_propagate = states[-1].a
		
		for l_id, l in reversed(list(enumerate(net))):
			if l_id == len(net)-1:
				continue
			states[l_id] = l.feedback(states[l_id], to_propagate)

		error = tf.nn.l2_loss(debug[-1].reconstruction - y)
		errors.append(error)

		# states[0] = (tf.Print(states[0][0], [error]),) + states[0][1:]

		states_acc.append(
			tuple(
				tuple(tf.identity(ss) for ss in s)
				for s in states
			)
		)
		
		debug_acc.append(
			tuple(
				tuple(tf.identity(rrt) for rrt in rr)
				for rr in debug
			)
		)



	error_rate = tf.reduce_mean(tf.cast(
		tf.not_equal(
			tf.argmax(debug[-1].reconstruction, axis=1),  
			tf.cast(tf.argmax(y, axis=1), tf.int64)
		), tf.float32))
	
	final_states = tuple(l.final_state(ls) for l, ls in zip(net, states))

	optimizer = tf.train.GradientDescentOptimizer(config.lrate)
	
	grads_and_vars = tuple(
		g_v
		for l_id, (l, s) in enumerate(zip(net, final_states))
	  	for g_v in l.get_grads_and_vars(s, x if l_id == 0 else states[l_id-1].a)
	)


	apply_grads_step = tf.group(
	    optimizer.apply_gradients(grads_and_vars),
	)

	return (
		GraphTuple(
			x, 
			y, 
			is_train,
			net, 
			final_states, 
			error_rate,
			apply_grads_step
		), 
		GraphDebugTuple(
			errors,
			states_acc, 
			debug_acc,
			grads_and_vars, 
		),
	)
