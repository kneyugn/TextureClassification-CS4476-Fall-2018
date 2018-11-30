from os import listdir
import os
from matplotlib.image import imread
from random import shuffle
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn import svm, metrics, datasets
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from LBP import getFeature
import learningCurve
from sklearn.preprocessing import StandardScaler

"""Plitting data100x100"""

allFolders = ['data100x100/' + dI for dI in os.listdir('data100x100') if os.path.isdir(os.path.join('data100x100', dI))]
paths = allFolders
allFiles = []
testNames = []
trainNames = []

for folderIndex, folder in enumerate(allFolders):
    onlyfiles = [f for f in listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for index,val in enumerate(onlyfiles):
        onlyfiles[index] = paths[folderIndex] + '/' + val
    #seventy = int(len(onlyfiles) * .7)
    trainNames += onlyfiles[:]
    #testNames += onlyfiles[seventy:]

trainImages = []
trainLabels = []
d = [1]
for path in trainNames:
    img = imread(path)
    feature = getFeature(path).reshape(-1)
    trainImages.append(feature)
    trainLabels.append(path.split('/')[1])
trainImages = np.array(trainImages)
print trainImages.shape


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(trainImages, trainLabels, test_size=0.2, random_state=0)

ker = 'poly'

expected = y_test

svc = svm.SVC(C=1.0,  class_weight=None, coef0=5,
    decision_function_shape='ovo', degree=7, gamma='auto', kernel=ker,
    max_iter=-1, probability=True, random_state=None, shrinking=True)
#model.fit(trainImages, trainLabels.tolist())
model = svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
#predicted = model.predict(testImages)

print("Classification report for classifier %s:\n%s\n"
      % (svc, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

precision = precision_score(expected, predicted, average='weighted')
recall = recall_score(expected, predicted, average='weighted')
accuracy = accuracy_score(expected, predicted)
print("precision", precision)
print("recall", recall)
print("accuracy", accuracy)

plt.scatter(y_test, predicted)
plt.xlabel('Expected Values')
plt.ylabel('Predicted Values')
plt.show()

