import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

logreg = LogisticRegression(lr = 0.001, n_iters = 1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("LR classification accuracy: ", accuracy(y_test, y_pred))