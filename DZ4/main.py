import numpy as np

# pip install liac-arff
import arff

import pandas as pan
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .0003  # step size in the mesh

names = ["Nearest Neighbors", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "Gaussian Analysis"]
classifiers = [
    KNeighborsClassifier(15),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    GaussianNB()]

file_name = "Training Dataset.arff"
data = arff.load(open(file_name, 'rb'))

output = open("output.txt", 'w')

y = list()
X = list()

for line in data['data']:
    float_line = list()
    for case in line:
        float_line.append(float(case))

    if float_line[-1] == -1:
        y.append(0)
    else:
        y.append(1)

    float_line.pop()
    X.append(float_line)

X = np.array(X)

separable = (X, y)

ds = separable

# preprocess dataset, split into training and test part
X, y = ds
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    output.write("Score of {0}:\n{1}\n\n".format(str(name), str(score)))

new_X_test = list()
for x in X_test:
    proba = ((classifiers[0].predict_proba(x) - 0.5) + (classifiers[1].predict_proba(x) - 0.5) +
    (classifiers[2].predict_proba(x) - 0.5) + (classifiers[3].predict_proba(x) - 0.5))
    new_X_test.append(proba)

i = 0
true = 0
all = 0
for x in new_X_test:
    if x[0][0] >= 0.5 and y_test[i] == 0:
        true += 1
    if x[0][0] < 0.5 and y_test[i] == 1:
        true += 1
    all += 1
    i += 1

output.write("Score of ensemble:\n{0}".format(str(float(true)/all)))