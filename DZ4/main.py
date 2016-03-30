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

Coef = [10, 10, 10, 10]
i = 0
true = 0
all = 0
ef = 3.0
new_X_test = list()
for x in X_test:
    Class = [classifiers[0].predict_proba(x), classifiers[1].predict_proba(x), classifiers[2].predict_proba(x), classifiers[3].predict_proba(x)]
    proba = (Coef[0]*(Class[0] - 0.7) + Coef[1]*(Class[1] - 0.7) + Coef[2]*(Class[2] - 0.3) + Coef[3]*(Class[3] - 0.4))
    new_X_test.append(proba)

    if y_test[i] == 0:
        if new_X_test[-1][0][0] >= 0:
            true += 1
        else:
            if Class[0][0][0] < 0:
                Coef[0] = Coef[0] / ef
            else:
                Coef[0] = Coef[0] * ef
            if Class[1][0][0] < 0:
                Coef[1] = Coef[1] / ef
            else:
                Coef[0] = Coef[0] * ef
            if Class[2][0][0] < 0:
                Coef[2] = Coef[2] / ef
            else:
                Coef[2] = Coef[2] * ef
            if Class[3][0][0] < 0:
                Coef[3] = Coef[3] / ef
            else:
                Coef[3] = Coef[3] * ef
    else:
        if new_X_test[-1][0][0] < 0:
            true += 1
        else:
            if Class[0][0][0] > 0:
                Coef[0] = Coef[0] / ef
            else:
                Coef[0] = Coef[0] * ef
            if Class[1][0][0] > 0:
                Coef[1] = Coef[1] / ef
            else:
                Coef[0] = Coef[0] * ef
            if Class[2][0][0] > 0:
                Coef[2] = Coef[2] / ef
            else:
                Coef[2] = Coef[2] * ef
            if Class[3][0][0] > 0:
                Coef[3] = Coef[3] / ef
            else:
                Coef[3] = Coef[3] * ef

    all += 1
    i += 1

output.write("Score of ensemble:\n{0}".format(str(float(true)/all)))
output.write("\nCoef = {0}".format(str(Coef)))