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
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

"""Plitting data"""

allFolders = ['full_data3/' + dI for dI in os.listdir('full_data3') if os.path.isdir(os.path.join('full_data3', dI))]
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
    glcm = greycomatrix(np.uint8(img), d, [0], 256, symmetric=True, normed=True)
    cont = greycoprops(glcm, 'contrast')[0, 0]
    diss = greycoprops(glcm, 'dissimilarity')[0, 0]
    homo = greycoprops(glcm, 'homogeneity')[0, 0]
    ener = greycoprops(glcm, 'energy')[0, 0]
    corr = greycoprops(glcm, 'correlation')[0, 0]
    asm = greycoprops(glcm, 'ASM')[0, 0]
    trainImages.append([cont, diss, homo, ener, corr, asm])
    trainLabels.append(path.split('/')[1])
trainImages = np.array(trainImages)

print 'middle check'

"""
testImages = []
testLabels = []
for path in testNames:
    img = imread(path)
    glcm = greycomatrix(np.uint8(img), d, [0], 256, symmetric=True, normed=True)
    cont = greycoprops(glcm, 'contrast')[0, 0]
    diss = greycoprops(glcm, 'dissimilarity')[0, 0]
    homo = greycoprops(glcm, 'homogeneity')[0, 0]
    ener = greycoprops(glcm, 'energy')[0, 0]
    corr = greycoprops(glcm, 'correlation')[0, 0]
    asm = greycoprops(glcm, 'ASM')[0, 0]
    testImages.append([cont, diss, homo, ener, corr, asm])
    testLabels.append(path.split('/')[1])
testImages = np.array(testImages)
trainImages = np.load('trainImg.npy')
trainLabels = np.load('trainLab.npy')
"""

#np.save('testImg', testImages)
#np.save('testLab', testLabels)

#np.save('trainImg100', trainImages)
#np.save('trainLab100', trainLabels)


np.savetxt('temp2.txt', trainImages)
np.savetxt('tempLabel2.txt', trainLabels, fmt="%s")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainImages, trainLabels, test_size=0.3, random_state=0)

print y_train
print y_test

expected = y_test

parameters = {
'C': [1, 2],
'kernel': ["linear", "poly", "rbf", "sigmoid"],
'degree' : [6, 7],
'gamma':['auto'],
'coef0':[4.0, 5.0],
'shrinking': [True],
'max_iter':[-1.0, 100, 200],
'decision_function_shape':['ovo','ovr']
}
print 'start svc'
svc = svm.SVC(kernel='poly', C=1, degree=6, shrinking=True, max_iter=-1, decision_function_shape='ovo', coef0=5.0, gamma='auto')
#clf = GridSearchCV(svc, parameters, cv=2)
#np.save('gridsearchResult', clf)

#clf = np.load('gridsearchResult.npy')
#model.fit(trainImages, trainLabels.tolist())
print 'start fit'
model = svc.fit(X_train, y_train)

print 'start predict'
predicted = svc.predict(X_test)
#predicted = model.predict(testImages)

print(classification_report(expected, predicted))

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
plt.xticks(rotation=45)
plt.show()