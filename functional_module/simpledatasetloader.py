import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data     = []
        labels   = []

        for (i, imagePath) in enumerate(imagePaths):
            """
            Load image and extract the class label.
            
            Assuming image path has the following format:
            /path/to/dataset/{class}/{imagename}.jpg
            """
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(labels)

            # show process info
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))


        print("[INFO] Data process end!")
        return (np.array(data), np.array(labels))