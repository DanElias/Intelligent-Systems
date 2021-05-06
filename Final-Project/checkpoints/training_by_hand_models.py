"""
Training made by hand models
Author: DanElias
Date: May 2021
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Look for class in upper directory
sys.path.append("..")
from models.logistic_regression import LogisticRegression
from models.decision_tree import DecisionTree

"""# Load Data"""
df = pd.read_csv("../data/kelloggs_reviews_labelled31-03.csv", encoding="UTF-8")

"""Split in Test and Train"""
x = df['text']
y = []


for i in range(len(df['label'])):
    if df['label'][i] == "i":
        y.append(1)
    else:
        y.append(2)
y = np.array(y)
print(y.dtype)
# split x and y into training and testing sets
# stratify returns training and test subsets that have the same proportions of class labels as the input dataset.
# each set contains approximately the same percentage of samples of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify = y)

# Using the best hyperparams for Logistic Regression

# TF-IDF Vectorization
vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, norm= 'l2', binary = True, analyzer = 'word')
x_train_dtm = vect.fit_transform(X_train)
x_test_dtm = vect.transform(X_test)

# ---- Logistic Regression by hand ----

"""
# Train
logreg = LogisticRegression(lr = 0.001, n_iters = 200)
logreg.fit(x_train_dtm, y_train)
y_pred = logreg.predict(x_test_dtm)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("LR classification accuracy: ", accuracy(y_test, y_pred))
"""

"""
Sigmoid function:
return 1 / (1 + np.exp(-arr))
TypeError: loop of ufunc does not support argument 0 of type csr_matrix which has no callable exp method
"""

"""
# ---- Decision Tree by hand ----

# Train
d_tree = DecisionTree()
d_tree.fit(x_train_dtm, y_train)
y_pred = d_tree.predict(x_test_dtm)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("DT classification accuracy: ", accuracy(y_test, y_pred))
"""