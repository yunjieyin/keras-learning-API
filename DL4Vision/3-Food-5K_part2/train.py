from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from Config import config
import numpy as np
import pickle
import os

def load_data_split(splitPath):
    data = []
    labels = []

    for row in open(splitPath):
        row = row.strip().split(',')
        label = row[0]
        features = np.array(row[1:], dtype='float')

        data.append(features)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)

trainingPath = os.path.sep.join([config.BASE_CSV_PATH, '{}.csv'.format(config.TRAIN)])
testingPath = os.path.sep.join([config.BASE_CSV_PATH, '{}.csv'.format(config.TEST)])

print('[INFO] loading data...')
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)

le = pickle.loads(open(config.LE_PATH, 'rb').read())

print('[INFO] training model...')
model = LogisticRegression(solver='lbfgs', multi_class='auto')
model.fit(trainX, trainY)

print('[INFO] evaluating...')
preds = model.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

print('[INFO] saving model...')
f = open(config.MODEL_PATH, 'wb')
f.write(pickle.dumps(model))
f.close()