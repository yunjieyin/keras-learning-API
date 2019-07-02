"""
Run environment:
    1. ubuntu
    2. tensorflow 1.10.0
    3. cuda9.0 cudnn7.3.1
"""

# USAGE
# 1.train from scratch:
#   python train.py --checkpoints ./ckpt
# 2.train with exist model
#   python train.py --checkpoints ckpt --model ckpt/epoch_55.hdf5 --start-epoch 55

import matplotlib
matplotlib.use("Agg")
import tensorflow as tf

from Config import emotion_config as config
from utils.preprocessing import ImageToArrayPreprocessor
from utils.callbacks import EpochCheckpoint
from utils.callbacks import TrainingMonitor
from utils.io import HDF5DatasetGenerator
from utils.nn.conv import EmotionVGGNet

import argparse
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Adam = tf.keras.optimizers.Adam
K = tf.keras.backend

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to model to be load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(rotation_range=10,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              rescale=1/255.0,
                              fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1/255.0)
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                               aug=trainAug, preprocessors=[iap],
                               classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
                              aug=valAug, preprocessors=[iap],
                              classes=config.NUM_CLASSES)


if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

else:
    print("[INFO] loading {}...".format(args["model"]))
    model = tf.keras.models.load_model(args["model"])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)

    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])

callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"])
]

model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages//config.BATCH_SIZE,
                    epochs=15,
                    max_queue_size=config.BATCH_SIZE * 2,
                    callbacks=callbacks,
                    verbose=1)

trainGen.close()
valGen.close()



