# import the necessary packages
import tensorflow as tf
# from keras.preprocessing.image import img_to_array
img_to_array = tf.keras.preprocessing.image.img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)