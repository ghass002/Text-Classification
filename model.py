from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D


def get_last_layer_units_and_activation(num_classes):

	if num_classes == 2:
		activation = 'sigmoid'
		units = 1
	else:
		activation = 'softmax'
		units = num_classes

	return units, activation

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
	"""creates an instance of multi layer percepton model
	"""
	op_units, op_activation = get_last_layer_units_and_activation(num_classes)
	model = models.Sequential()
	model.add(Dropout(rate = dropout_rate, input_shape = input_shape))

	for _ in range(layers-1):
		model.add(Dense(units = units, activation = 'relu'))
		model.add(Dropout(rate = dropout_rate))

	model.add(Dense(units = op_units, activation = op_activation))
	return model

def sepcnn_model(blocks, filter, kernel_size, embedding_dim, dropout_rate, pool_size, input_shape, num_classes, num_features, use_pretrained_embedding=False, is_embedding_trainable= False, embedding_matrix = None):
""" create a separable CNN model
# Arguments
	blocks: int number of pairs of seppCNN and pooling blocks in the model
	filters: int, output dimension of the layers
	kernel_size: int, length of the convolution window
	embedding_dim: int, dimension of the embedding vector
	dropout_rate: float, percentage of input to drop at dropout layers
	pool_size: int, factor by which to downscale input at maxpooling layer
	input_shape: tuple, shape of input 
	num_classes: int, number of classes
	num_features: int, number of words ( embedding input dimension )
	use_pretrained_embedding: bool true if pre trained embedding is on
	is_embedding_trainable: bool, true if embedding layer is trainable
	embedding_matrix: dict, dictionary with embedding coefficients. 

	"""
	op_units, op_activation = get_last_layer_units_and_activation(num_classes)
	model = models.Sequential()

	#Add embedding layer. If pretrained weights is used add weights to the embeddings layer and set trainable to is_embedding_trainable flag
	if use_pretrained_embedding:
		model.add(input_dim = num_features, output_dim = embedding_dim, input_length = input_shape[0], weights = [embedding_matrix], trainable = is_embedding_trainable)
	else:
		model.add(Embedding(input_dim = num_features, output_dim = embedding_dim, input_length = input_shape[0]))

	for _ in range(blocks - 1):
		model.add(Dropout(rate = dropout_rate))
		model.add(SeparableConv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', bias_initializer = 'random_uniform', depthwise_initializer= 'random_uniform, padding = 'same'))
		model.add(SeparableConv1D(filters = filters,  kernel_size = kernel_size, activation = 'relu', bias_initializer = 'random_uniform', padding = 'same'))			
		model.add(MaxPooling1D(pool_size=pool_size))
		

	model.add(SeparableConv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', bias_initializer='random_uniform', depthwise_initializer='random_uniform', padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,  kernel_size=kernel_size,  activation='relu', bias_initializer='random_uniform', depthwise_initializer='random_uniform', padding='same'))
    model.add(GlobalAveragePooling1D())
   	model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
	
	return model