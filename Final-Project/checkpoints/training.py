# -*- coding: utf-8 -*-
"""
app.py: Main program that receives new queries to be classified by chosen models
Author: DanElias
Date: May 2021
"""

"""### Import python libraries"""

import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Natural Language Processing tools

"""
Lemmatization is the process of grouping together the different inflected forms
of a word so they can be analysed as a single item. Lemmatization is similar to
stemming but it brings context to the words. So it links words with similar
meaning to one word. car, cars, car's, cars' ->  car
Stemming just removes the last few characters, often leading to incorrect
meanings and spelling errors
https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

"""
from nltk.stem import WordNetLemmatizer

"""
Convert a collection of text documents to a matrix of token counts
This implementation produces a sparse representation of the counts
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
"""
from sklearn.feature_extraction.text import CountVectorizer

"""
Term Frequency - Inverse Document Frequency
Measure of orginiality of a word by comparing the times a word appears in a doc
with the number of doc the word appears in
Convert a collection of raw documents to a matrix of TF-IDF features.
Equivalent to CountVectorizer followed by TfidfTransformer.
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
"""
from sklearn.feature_extraction.text import TfidfVectorizer


# Machine Learning Models - Classifiers

"""
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
"""
from sklearn.naive_bayes import MultinomialNB

"""
Logistic Regression (aka logit, MaxEnt) classifier.
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""
from sklearn.linear_model import LogisticRegression

"""
Classifier implementing the k-nearest neighbors vote.
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
from sklearn.neighbors import KNeighborsClassifier

"""
A decision tree classifier.
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""
from sklearn.tree import DecisionTreeClassifier


"""
Linear classifiers (SVM, logistic regression, etc.) with SGD training
This estimator implements regularized linear models with stochastic gradient
descent (SGD) learning: the gradient of the loss is estimated each sample at a
time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
By default, it fits a linear support vector machine (SVM).
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
"""
from sklearn.linear_model import SGDClassifier

"""
Linear Support Vector Classification.
SVM or Support Vector Machine is a linear model for classification
and regression problems.
The idea of SVM is simple: The algorithm creates a line or a hyperplane which
separates the data into classes.
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
"""
from sklearn.svm import LinearSVC

# Ensemble Methods

"""
A random forest is a meta estimator that fits a number of decision tree classifiers on various 
sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
GB builds an additive model in a forward stage-wise fashion; 
it allows for the optimization of arbitrary differentiable loss functions.
In each stage n_classes_ regression trees are fit on the negative gradient of
the binomial or multinomial deviance loss function. Binary classification is a
special case where only a single regression tree is induced.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

"""
A Bagging classifier is an ensemble meta-estimator that fits base classifiers
each on random subsets of the original dataset and then aggregate their
individual predictions (either by voting or by averaging) to form a
final prediction.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
"""
from sklearn.ensemble import BaggingClassifier

"""
An AdaBoost classifier is a meta-estimator that begins by fitting a classifier
on the original dataset and then fits additional copies of the classifier on the
same dataset but where the weights of incorrectly classified instances are
adjusted such that subsequent classifiers focus more on difficult cases.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
"""
from sklearn.ensemble import AdaBoostClassifier


# Tunning Tools

"""
Normalize samples individually to unit norm.
Each sample (i.e. each row of the data matrix) with at least one non zero
component is rescaled independently of other samples so that its norm
(l1, l2 or inf) equals one.
Scaling inputs to unit norms is a common operation for text classification or
clustering for instance.
For instance the dot product of two l2-normalized TF-IDF vectors is the cosine
similarity of the vectors and is the base similarity metric for the
Vector Space Modelcommonly used by the Information Retrieval community.
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
"""
from sklearn.preprocessing import Normalizer

"""
Pipeline of transforms with a final estimator.
Sequentially apply a list of transforms and a final estimator. Intermediate steps
of the pipeline must be ‘transforms’, that is, they must implement fit and
transform methods. The final estimator only needs to implement fit.
The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
"""
from sklearn.pipeline import Pipeline

"""
Evaluate metric(s) by cross-validation and also record fit/score times.
The training set is split into k smaller sets
A model is trained using k-1 of the folds as training data;
the resulting model is validated on the remaining part of the data (it is used as a test set to compute a performance measure such as accuracy).
The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
This approach can be computationally expensive, but does not waste too much data
https://scikit-learn.org/stable/modules/cross_validation.html
"""
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score


"""
Randomized search on hyper parameters for each model.
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
"""
from sklearn.model_selection import RandomizedSearchCV

"""# Load Data"""

df = pd.read_csv("../data/kelloggs_reviews_labelled31-03.csv", encoding="UTF-8")
#df.head()
#df.shape
#df.label.value_counts()

"""# Cleaning"""

df['text'] = df['text'].str.lower() # Change to lower case
df['text'] = df['text'].apply(lambda x: np.str_(x)) # Remove trailing white spaces
df['text'] = df['text'].str.replace('\B|[^a-z\'\-? ]', '') # Remove special characters

"""Split in Test and Train"""

x = df['text']
y = df['label']

# split x and y into training and testing sets
# stratify returns training and test subsets that have the same proportions of class labels as the input dataset.
# each set contains approximately the same percentage of samples of each target class as the complete set.
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify = y)

# examine the object shapes
print(y_train.value_counts())
print(y_test.value_counts())

"""# Building the Vocabulary"""

vect = CountVectorizer()

# Transform documents to document-term matrix. Extract token counts
vect.fit(x_train)
x_train_dtm = vect.transform(x_train)

# only transform x_test
x_test_dtm = vect.transform(x_test)

# examine the shapes: rows are documents, columns are terms (aka "tokens" or "features")
print(x_train_dtm.shape)
print(x_test_dtm.shape)

# examine the last 50 features
print(vect.get_feature_names()[-50:])

"""# Comparing the accuracy of different approaches"""

vects = [CountVectorizer(),TfidfVectorizer()] # The two types of vectorizers
vectnames = ["Count Vect","Tfidf Vect"]

# The machine learning classifiers to be used and compared
clfs = [
        MultinomialNB(),
        LinearSVC(),
        LogisticRegression(),
        SGDClassifier(),
        RandomForestClassifier(),
        BaggingClassifier(RandomForestClassifier()), 
        GradientBoostingClassifier(), 
        AdaBoostClassifier()
        ]

clfnames = [
            "Multinomial Naive Bayes",
            "Linear SVM",
            "Logistic Regression",
            "Stochastic Gradient Descent",
            "Random Forest",
            "Bagging Random Forest",  
            "Gradient Boosting", 
            "Ada Boost"]

#building a pipeline

for vectname, vect, in zip(vectnames, vects):
    for clfname, clf in zip(clfnames, clfs):
        pipe = Pipeline([
            ('vect', vect),
            ('clf', clf),
        ])       

        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        train_acc = metrics.accuracy_score(y_train, pipe.predict(x_train))
        test_acc = metrics.accuracy_score(y_test, pred)
        print("{} + {} - train acc: {} test acc: {} ".format(vectname, clfname, train_acc, test_acc))

"""Best result = Tfidf Vect + Linear SVM - train acc: 0.9880763116057234 test acc: 0.7857142857142857"""

tfidf = TfidfVectorizer()
linear_svm = LinearSVC()

tfidf.fit(x_train)
x_train_dtm = tfidf.transform(x_train)
x_test_dtm = tfidf.transform(x_test)

linear_svm.fit(x_train_dtm, y_train)
y_pred = linear_svm.predict(x_test_dtm)
x_pred = linear_svm.predict(x_train_dtm)
train_acc = metrics.accuracy_score(y_train, x_pred)
test_acc = metrics.accuracy_score(y_test, y_pred)

print("{} + {} - train acc: {} test acc: {} ".format("Tfidf Vect", "Linear SVM", train_acc, test_acc))

linear_svm_vect = tfidf

# Hyperparameter Tunning for Linear SVM
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lsvc', LinearSVC())
])

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__analyzer':['char', 'word', 'char_wb'],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__binary': [True, False],
    'tfidf__norm': ['l2'], 
    'lsvc__penalty': ['l2'],
    'lsvc__loss': ['squared_hinge'],
    'lsvc__dual': [True, False],
    'lsvc__tol': [1e-4, 1e-3, 1e-2, 1e-1],
    'lsvc__C': np.logspace(1, 4, num=10),
    'lsvc__multi_class': ['ovr', 'crammer_singer'],
    'lsvc__max_iter': [100,200, 500]
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Linear SVC
vect = TfidfVectorizer(ngram_range=(1, 1), max_df=0.25, norm= 'l1', binary = True, analyzer = 'word')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

lsvc = LinearSVC(penalty='l2', multi_class='ovr', max_iter=200, loss='squared_hinge', dual=False, C=46.41)
lsvc.fit(x_train_dtm, y_train)
y_pred_class = lsvc.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, lsvc.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

# Hyperparameter Tunning for Logistic Regression
pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('lr', LogisticRegression())
])

parameters = {
    'cv__lowercase': [True, False],
    'cv__max_df': (0.25, 0.5, 0.75),
    'cv__analyzer':['char', 'word', 'char_wb'],
    'cv__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'cv__binary': [True, False],
    'lr__dual': [True,False],
    'lr__max_iter': [100,200,500,1000],
    'lr__C' :np.logspace(0, 4, num=10)
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Logistic Regression
vect = CountVectorizer(ngram_range=(1, 2), max_df=0.25, lowercase=True, binary = True, analyzer = 'word')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

lr = LogisticRegression(max_iter = 100, dual = False, C = 7.7426)
lr.fit(x_train_dtm, y_train)
y_pred_class = lr.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, lr.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

# Hyperparameter Tunning for Linear SVM CountVectorizer
pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('lsvc', LinearSVC())
])

parameters = {
    'cv__lowercase': [True, False],
    'cv__max_df': (0.25, 0.5, 0.75),
    'cv__analyzer':['char', 'word', 'char_wb'],
    'cv__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'cv__binary': [True, False],
    'lsvc__penalty': ['l1', 'l2'],
    'lsvc__loss': ['squared_hinge'],
    'lsvc__dual': [True, False],
    'lsvc__tol': [1e-4, 1e-3, 1e-2, 1e-1],
    'lsvc__C': np.logspace(1, 4, num=10),
    'lsvc__multi_class': ['ovr', 'crammer_singer'],
    'lsvc__max_iter': [100, 200, 500, 1000]
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Linear SVC
vect = CountVectorizer(ngram_range=(1, 2), max_df=0.75, lowercase=True, binary = True, analyzer = 'word')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

lsvc = LinearSVC(tol=0.001, penalty='l2', multi_class='crammer_singer', max_iter=200, loss='squared_hinge', dual=False, C=21.54)
lsvc.fit(x_train_dtm, y_train)
y_pred_class = lsvc.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, lsvc.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

# Hyperparameter Tunning for Logistic Regression
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__analyzer':['char', 'word', 'char_wb'],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__binary': [True, False],
    'tfidf__norm': [None, 'l1', 'l2'], 
    'lr__dual': [True,False],
    'lr__max_iter': [100,200, 500],
    'lr__C' :np.logspace(0, 4, num=10)
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Logistic Regression
vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, norm= 'l2', binary = True, analyzer = 'word')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

lr = LogisticRegression(max_iter = 200, dual = False, C = 10000)
lr.fit(x_train_dtm, y_train)
y_pred_class = lr.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, lr.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

log_reg = lr
log_reg_vect = vect

# Hyperparameter Tunning for M Naive Bayes
pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('nb', MultinomialNB())
])

parameters = {
    'cv__lowercase': [True, False],
    'cv__max_df': (0.25, 0.5, 0.75),
    'cv__analyzer':['char', 'word', 'char_wb'],
    'cv__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'cv__binary': [True, False],
    'nb__alpha': [1.0, 0],
    'nb__fit_prior': [True, False],
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Naive Bayes
vect = CountVectorizer(ngram_range=(1, 1), max_df=0.25, lowercase=True, binary=False, analyzer = 'word')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

mnb = MultinomialNB(fit_prior=True, alpha=1.0)
mnb.fit(x_train_dtm, y_train)
y_pred_class = mnb.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, mnb.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

mnaive_bayes = mnb
mnaive_bayes_vect = vect

# Hyperparameter Tunning for Stochastic Gradient Descent
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('sgd', SGDClassifier())
])

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__analyzer':['char', 'word', 'char_wb'],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__binary': [True, False],
    'tfidf__norm': [None, 'l1', 'l2'], 
    'sgd__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
    'sgd__max_iter': [5, 100, 500, 1000], # number of epochs
    'sgd__loss': ['log'],
    'sgd__penalty': ['l2', 'l1', 'elasticnet'], 
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
#roc_auc = metrics.roc_auc_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Stochastic Gradient Descent
vect = TfidfVectorizer(ngram_range=(1, 3), max_df=0.25, norm= 'l2', binary = False, analyzer = 'word')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

sgd = SGDClassifier(penalty = 'l1', max_iter = 1000, loss = 'log', alpha = 0.0001, random_state = 1)
sgd.fit(x_train_dtm, y_train)
y_pred_class = sgd.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, sgd.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

stochastic_gd = sgd
stochastic_gd_vect = vect

# Hyperparameter Tunning for Random Forest
pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('rf', RandomForestClassifier())
])

parameters = {
    'cv__lowercase': [True, False],
    'cv__max_df': (0.25, 0.5, 0.75),
    'cv__analyzer':['char', 'word', 'char_wb'],
    'cv__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'cv__binary': [True, False],
    'rf__n_estimators': [100,200,300,500],
    'rf__criterion': ['gini', 'entropy'],
    'rf__min_samples_split': [2,4,8,16],
    'rf__min_samples_leaf': [1,2,5,10],
    'rf__bootstrap': [True, False],
    'rf__oob_score': [True, False],
    'rf__random_state': [42,1,10],
    'rf__warm_start': [True, False]
}

random =RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=2)

start_time = time.time()
random_result = random.fit(x_train, y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

pred = random.predict(x_test)
test_acc = metrics.accuracy_score(y_test, pred)
pre_macro = metrics.precision_score(y_test, pred, average="macro")
recall_macro = metrics.recall_score(y_test, pred, average="macro")
f1_macro = metrics.f1_score(y_test, pred, average="macro")

print("test acc: {} recall: {} f1: {}".format(test_acc, recall_macro, f1_macro))

# Using the best hyperparams for Random Forest
vect = CountVectorizer(ngram_range=(1, 3), max_df=0.5, lowercase=True, binary = True, analyzer = 'char_wb')
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

rf = RandomForestClassifier(warm_start=False,random_state=42,oob_score=False,n_estimators=500, min_samples_split=8, min_samples_leaf=10,criterion='entropy',bootstrap=False)
rf.fit(x_train_dtm, y_train)
y_pred_class = rf.predict(x_test_dtm)

# Get the training accuracy
print('Training Accuracy: ', metrics.accuracy_score(y_train, rf.predict(x_train_dtm)))
# print the accuracy of its predictions
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))

random_forest = rf
random_forest_vect = vect

"""# Reccurent Neural Network"""

# Import necessary libraries
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional
from keras.models import Model
from keras.models import Sequential

df["text"] = df["text"].astype(str)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#tokenizer.fit_on_texts(reviews.Review_clean.values)

tokenizer.fit_on_texts(df["text"].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

from keras.preprocessing.sequence import pad_sequences
# Truncate and pad the input sequences so that they are all in the same length for modeling

X = tokenizer.texts_to_sequences(df["text"].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df["label"]).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

from keras.layers import Dense, Embedding, LSTM, Dropout

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr_test = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr_test[0],accr_test[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();

df2 = pd.DataFrame(x_test)
df2.head()

ans = []
for i in df2["text"]:
    new_review = [i]
    seq = tokenizer.texts_to_sequences(new_review)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['e', 'i']
    ans.append(labels[np.argmax(pred)])

df2['Predicted'] = ans
df2['Actual']=y_test

#df2.head(-10)

confusion_matrix = pd.crosstab(df2['Actual'], df2['Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
# 144 True positives, 256 True negatives, 12 false positives, 10 false negatives

sn.set(font_scale=1.5) # for label size
sn.heatmap(confusion_matrix, fmt='g', annot=True, annot_kws={"size": 15}) # font size
plt.show()

"""# Using the models to predict new data queries

### Chosen Models
"""

# Recurrent Neural Network - Keras
# TFIDF + Linear SVM
# TFIDF + Logistic Regression - Tuned
# CV + Multinomial Naive Bayes - Tuned
# TFIDF + Stochastic Gradient Descent - Tuned
# CV + Random Forest - Tuned

chosen_models = {
    "Linear SVM": (linear_svm, linear_svm_vect),
    "Logistic Regression": (log_reg, log_reg_vect),
    "Naive Bayes": (mnaive_bayes, mnaive_bayes_vect),
    "Stochastic Gradient Descent": (stochastic_gd, stochastic_gd_vect),
    "Random Forest": (random_forest, random_forest_vect)}

"""# Serialize the models"""

# Neural Network Serialization
# for heavy model architectures, .h5 file is unsupported.
from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("../test_folder_models/neural_network.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../test_folder_models/neural_network.h5")
print("Saved model to disk")

# Save the dictionary of models in pickle file
import pickle
with open('../test_folder_models/models.pickle', 'wb') as handle:
    pickle.dump(chosen_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

import pickle
with open('../test_folder_models/neural_network_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)