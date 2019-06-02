from Config import config
from imutils import paths
import shutil
import os

for split in (config.TRAIN, config.TEST, config.VAL):
    print('[INFO] processing {} split...'.format(split))

    p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
    imagePaths = list(paths.list_images(p))

    for imagePath in imagePaths:
        filename = imagePath.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split('_')[0])]

        dirPath = os.path.sep.join([config.BASE_PATH, split, label])
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, p)
