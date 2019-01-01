# USAGE
# python train.py --dataset /media/yinyunjie/WD_HDD/dataSets/101_ObjectCategories


import matplotlib
matplotlib.use("Agg")

from CNNs.stridednet import StridedNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="# of epochs to train network")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

LABELS = set(["Faces", "Leopards", "Motorbikes", "airplanes"])

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    if label not in LABELS:
        continue

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# split data: train,75%  test,25%
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, stratify=labels, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

print("[INFO] compiling model...")
opt = Adam(lr=1e-4, decay=1e-4 / args["epochs"])
model = StridedNet.build(width=96, height=96, depth=3,
    classes=len(lb.classes_), reg=l2(0.0005))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network for {} epochs...".format(args["epochs"]))
H = model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
    epochs=args["epochs"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# plot the train loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])