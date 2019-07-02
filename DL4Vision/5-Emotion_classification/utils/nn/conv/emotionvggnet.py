# import the necessary packages
import tensorflow as tf

# from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.layers.advanced_activations import ELU
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dropout
# from keras.layers.core import Dense
# from keras import backend as K

Sequential = tf.keras.models.Sequential
l = tf.keras.layers
K = tf.keras.backend

class EmotionVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Block #1: first CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(l.Conv2D(32, (3, 3), padding="same",
			kernel_initializer="he_normal", input_shape=inputShape))
		model.add(l.ELU())
		model.add(l.BatchNormalization(axis=chanDim))
		model.add(l.Conv2D(32, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(l.ELU())
		model.add(l.BatchNormalization(axis=chanDim))
		model.add(l.MaxPooling2D(pool_size=(2, 2)))
		model.add(l.Dropout(0.25))

		# Block #2: second CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(l.Conv2D(64, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(l.ELU())
		model.add(l.BatchNormalization(axis=chanDim))
		model.add(l.Conv2D(64, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(l.ELU())
		model.add(l.BatchNormalization(axis=chanDim))
		model.add(l.MaxPooling2D(pool_size=(2, 2)))
		model.add(l.Dropout(0.25))

		# Block #3: third CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(l.Conv2D(128, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(l.ELU())
		model.add(l.BatchNormalization(axis=chanDim))
		model.add(l.Conv2D(128, (3, 3), kernel_initializer="he_normal",
			padding="same"))
		model.add(l.ELU())
		model.add(l.BatchNormalization(axis=chanDim))
		model.add(l.MaxPooling2D(pool_size=(2, 2)))
		model.add(l.Dropout(0.25))

		# Block #4: first set of FC => RELU layers
		model.add(l.Flatten())
		model.add(l.Dense(64, kernel_initializer="he_normal"))
		model.add(l.ELU())
		model.add(l.BatchNormalization())
		model.add(l.Dropout(0.5))

		# Block #6: second set of FC => RELU layers
		model.add(l.Dense(64, kernel_initializer="he_normal"))
		model.add(l.ELU())
		model.add(l.BatchNormalization())
		model.add(l.Dropout(0.5))

		# Block #7: softmax classifier
		model.add(l.Dense(classes, kernel_initializer="he_normal"))
		model.add(l.Activation("softmax"))

		# return the constructed network architecture
		return model

if __name__ == "__main__":
	# visualize the network architecture
	from keras.utils import plot_model
	from keras.regularizers import l2
	model = EmotionVGGNet.build(48, 48, 1, 6)
	plot_model(model, to_file="model.png", show_shapes=True,
		show_layer_names=True)