from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

def get_subpixel_net(downsampleFactor, net_block, net_layers, channels=2):
	# initialize an input layer
	inputs = Input((None, None, 1))
	x = Conv1D(512, 7, padding="same", activation="LeakyReLU", kernel_initializer="Orthogonal")(inputs)
	x = Conv1D(512, 5, padding="same", activation="LeakyReLU", kernel_initializer="Orthogonal")(x)
	x = Conv1D(256, 3, padding="same", activation="LeakyReLU", kernel_initializer="Orthogonal")(x)
	x = net_block(x, numLayers=net_layers)
    # pass the inputs through a final CONV layer such that the
	# channels of the outputs can be spatially organized into
	# the output resolution
	x = Conv1D(channels * (downsampleFactor ** 2), 3, padding="same", activation="LeakyReLU", kernel_initializer="Orthogonal")(x)
	outputs = tf.nn.depth_to_space(x, downsampleFactor)
	# construct the final model and return it
	model = Model(inputs, outputs)
	return model


# in pixel shuffle we are rearranging the elements of H x W x ( C * r * r) to (H*r) x (W*r) x C