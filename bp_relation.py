
from model import ClassificationLayerNonRec

import tensorflow as tf
import numpy as np
from util import *


from graph import build_graph
from model import StateTuple, ReconstructionLayerNonRec, ClassificationLayerNonRec


input_size = 20

np.random.seed(43)
tf.set_random_seed(43)


x_v, target_v = get_toy_data(input_size, 2000)
y_v = one_hot_encode(target_v)
output_size = y_v.shape[1]

layer_size = 50

config = Config(
	tau = 5.0,
	tau_gamma = 1.0,
	tau_m = 1000.0,
	adapt_gain = 1000.0,
	h = 0.2,
	gamma = 5.0,
	num_iter = 30,
	lrate=1e-01,
)

gt, gdt = build_graph(
	input_size, 
	net_structure = (layer_size, output_size), 
	config = config, 
	net_neuron = ReconstructionLayerNonRec, 
	out_neuron = ClassificationLayerNonRec
)
