"""
Logistic Regression by Hand
Author: DanElias
Reference: https://www.youtube.com/watch?v=JDU3AzH3WKg&ab_channel=PythonEngineer
Date: May 2021
"""
# Using sigmoid function to get values between 0 and 1
# We need to calculate the weight and the bias
# Gradient Descent
# We use a cost function - Cross Entropy

# Update rules
# w =  w - a (dot) dw
# b = b - a (dot) db

import numpy as np

class LogisticRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        X is a numpy vector of size m x n, where m samples and n feature
        y is a numpy vector of m length
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        # Apply linear model and then sigmoid function
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            # derivatives to update
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, X):
        numlist = []
        for i in range(X.shape[0]):
            x = X[i].astype(np.float64)
            numlist.append(x)
        arr = np.array(numlist)
        return 1 / (1 + np.exp(-arr))

