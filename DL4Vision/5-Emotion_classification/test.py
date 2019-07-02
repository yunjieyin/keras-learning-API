# USAGE
# python test.py --model ckpt/epoch_70.hdf5

from Config import emotion_config as config
from utils.preprocessing import ImageToArrayPreprocessor
from utils.io import HDF5DatasetGenerator
import argparse
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model to load")
args = vars(ap.parse_args())

testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(config.TEST_HDF5,
                               config.BATCH_SIZE,
                               aug=testAug,
                               preprocessors=[iap],
                               classes=config.NUM_CLASSES)

print("[INFO] loading {}...".format(args["model"]))
model = tf.keras.models.load_model(args["model"])

(loss, acc) = model.evaluate_generator(
	testGen.generator(),
	steps=testGen.numImages // config.BATCH_SIZE,
	max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

# close the testing database
testGen.close()
