"""
app.py: Main program that receives new queries to be classified by chosen models
Author: DanElias
Date: May 2021
"""

#  Python libraries
import math
import time
import pickle
import os
import warnings

# Data manipulation
import numpy as np
import pandas as pd

# Data Vizualization libraries
import matplotlib.pyplot as plt
import seaborn as sn

# Sklearn tools
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Natural Language Processing tools
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning Models - Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Ensemble Methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Tunning Tools
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Keras
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

# Disable TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable Sklearn warning
def warn(*args, **kwargs):
    pass
warnings.warn = warn


class App:
    def __init__(self):
        # Constants for the Neural Network Tokenizer
        # The maximum number of words to be used. (most frequent)
        self.MAX_NB_WORDS = 50000
        # Max number of words in each complaint.
        self.MAX_SEQUENCE_LENGTH = 250
        # This is fixed.
        self.EMBEDDING_DIM = 100
        self.loaded_neural_network_model = self.load_neural_network()
        self.tokenizer = self.load_neural_network_tokenizer()
        self.loaded_models = self.load_models()
        self.examples()
        self.run()
        
    def run(self):
        """
        Recurrent Neural Network - Keras
        TFIDF + Linear SVM
        TFIDF + Logistic Regression - Tuned
        CV + Multinomial Naive Bayes - Tuned
        TFIDF + Stochastic Gradient Descent - Tuned
        CV + Random Forest - Tuned
        """
        close = False
        switcher = {
            0: "zero",
            1: "one",
            2: "two",
        }
        while(not close):
            print("\n\t--- REVIEWS CLASSIFICATION ---")
            new_review = input("\nWrite the new review to be classified: ")
            print("\n*** Available Classification Models ***")
            print("\n\ta) Neural Network with Keras")
            print("\n\tb) TFIDF + Linear SVM")
            print("\n\tc) TFIDF + Logistic Regression")
            print("\n\td) CV + Multinomial Naive Bayes")
            print("\n\te) TFIDF + Stochastic Gradient Descent")
            print("\n\tf) CV + Random Forest")
            print("\n\tg) Exit")
            option = input("\n\nType in the letter of your selected option:")

            if option == 'a':
                self.rnn(new_review)
            elif option == 'b':
                self.svm(new_review)
            elif option == 'c':
                self.logreg(new_review)
            elif option == 'd':
                self.nb(new_review)
            elif option == 'e':
                self.sgd(new_review)
            elif option == 'f':
                self.rf(new_review)
            elif option == 'g':
                print("\nSee you soon!")
                close = True
            else:
                print("\nInvalid option, make another selection")

    # Models
     
    def rnn(self, review = ""):
        print("\nYou have chosen: Neural Network with Keras")
        labels = ['e', 'i']
        new_review = [review]
        seq = self.tokenizer.texts_to_sequences(new_review)
        padded_review = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        pred = self.loaded_neural_network_model.predict(padded_review)
        classification = labels[np.argmax(pred)]
        print(pred, classification)
        if classification == 'i':
            print("\nThe review has been classified as: Intrinsic")
        else:
            print("\nThe review has been classified as: Extrinsic")
        print("\nThis model has 80% accuracy")

    def svm(self, review = ""):
        print("\nYou have chosen: TFIDF + Linear SVM")
        chosen_model = self.loaded_models["Linear SVM"][0]
        chosen_vect = self.loaded_models["Linear SVM"][1]
        x_query_dtm = chosen_vect.transform([review])
        y_pred = chosen_model.predict(x_query_dtm)
        labels = ['i', 'e']
        classification = labels[np.argmax(y_pred)]
        print(y_pred, classification)
        if classification == 'i':
            print("\nThe review has been classified as: Intrinsic")
        else:
            print("\nThe review has been classified as: Extrinsic")
        print("\n This model has 78.5% accuracy")


    def abstract_model_query(self, review, model_name):
        """
        Query sequence for simple sklearn models
        """
        labels = ['e', 'i']
        chosen_model = self.loaded_models[model_name][0]
        chosen_vect = self.loaded_models[model_name][1]
        x_query_dtm = chosen_vect.transform([review])
        y_pred = chosen_model.predict_proba(x_query_dtm)
        classification = labels[np.argmax(y_pred)]
        print(y_pred, classification)
        if classification == 'i':
            print("\nThe review has been classified as: Intrinsic")
        else:
            print("\nThe review has been classified as: Extrinsic")


    def logreg(self, review = ""):
        print("\nYou have chosen: TFIDF + Logistic Regression")
        self.abstract_model_query(review, "Logistic Regression")
        print("\n This model has 78% accuracy")


    def nb(self, review = ""):
        print("\nYou have chosen: CV + Multinomial Naive Bayes")
        self.abstract_model_query(review, "Naive Bayes")
        print("\n This model has 77.3% accuracy")


    def sgd(self, review = ""):
        print("\nYou have chosen: TFIDF + Stochastic Gradient Descent")
        self.abstract_model_query(review, "Stochastic Gradient Descent")
        print("\n This model has 77% accuracy")


    def rf(self, review = ""):
        print("\nYou have chosen: CV + Random Forest")
        self.abstract_model_query(review, "Random Forest")
        print("\n This model has 75% accuracy")

    # Loading the models

    def load_neural_network(self):
        json_file = open('models/neural_network.json', 'r')
        loaded_neural_network_json = json_file.read()
        json_file.close()
        loaded_neural_network_model = model_from_json(loaded_neural_network_json)
        # load weights into new model
        loaded_neural_network_model.load_weights("models/neural_network.h5")
        return loaded_neural_network_model


    def load_neural_network_tokenizer(self): 
        # load pickle tokenizer for neural network
        with open('models/neural_network_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer


    def load_models(self):
        # load pickle models dictionary
        with open('models/models.pickle', 'rb') as handle:
            loaded_models = pickle.load(handle)
        return loaded_models

    # Examples

    def examples(self):
        labels = ['e', 'i']
        new_intrinsic_review = ['I love the taste of my Special K']
        seq = self.tokenizer.texts_to_sequences(new_intrinsic_review)
        padded_intrinsic_review = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)

        new_extrinsic_review = ['I hate the branding of Kellogg\'s']
        seq = self.tokenizer.texts_to_sequences(new_extrinsic_review)
        padded_extrinsic_review = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)

        """
        Example Queries with Neural Network
        """
        print("\n--- Example Queries with Neural Network ---")
        print(new_intrinsic_review)
        pred = self.loaded_neural_network_model.predict(padded_intrinsic_review)
        print(pred, labels[np.argmax(pred)])

        print(new_extrinsic_review)
        pred = self.loaded_neural_network_model.predict(padded_extrinsic_review)
        print(pred, labels[np.argmax(pred)])

        """
        Example Queries with TFIDF + Linear SVM
        """
        print("\n--- Example Queries withTFIDF + Linear SVM ---")
        print(new_extrinsic_review)
        chosen_model = self.loaded_models["Linear SVM"][0]
        chosen_vect = self.loaded_models["Linear SVM"][1]
        new_review = new_extrinsic_review
        x_query_dtm = chosen_vect.transform(new_review)
        y_pred = chosen_model.predict(x_query_dtm)
        labels2 = ['i', 'e']
        print(y_pred, labels2[np.argmax(y_pred)])

        """
        Example Queries with TFIDF + Logistic Regression - Tuned
        """
        print("\n--- Example Queries with TFIDF + Logistic Regression - Tuned ---")
        print(new_extrinsic_review)
        chosen_model = self.loaded_models["Logistic Regression"][0]
        chosen_vect = self.loaded_models["Logistic Regression"][1]
        new_review = new_extrinsic_review
        x_query_dtm = chosen_vect.transform(new_review)
        y_pred = chosen_model.predict_proba(x_query_dtm)
        print(y_pred, labels[np.argmax(y_pred)])

        """
        Example Queries with CV + Multinomial Naive Bayes - Tuned
        """
        print("\n--- Example Queries with CV + Multinomial Naive Bayes - Tuned ---")
        print(new_extrinsic_review)
        chosen_model = self.loaded_models["Naive Bayes"][0]
        chosen_vect = self.loaded_models["Naive Bayes"][1]
        new_review = new_extrinsic_review
        x_query_dtm = chosen_vect.transform(new_review)
        y_pred = chosen_model.predict_proba(x_query_dtm)
        print(y_pred, labels[np.argmax(y_pred)])

        """
        Example Queries with TFIDF + Stochastic Gradient Descent - Tuned
        """
        print("\n--- Example Queries with TFIDF + Stochastic Gradient Descent - Tuned ---")
        print(new_extrinsic_review)
        chosen_model = self.loaded_models["Stochastic Gradient Descent"][0]
        chosen_vect = self.loaded_models["Stochastic Gradient Descent"][1]
        new_review = new_extrinsic_review
        x_query_dtm = chosen_vect.transform(new_review)
        y_pred = chosen_model.predict_proba(x_query_dtm)
        print(y_pred, labels[np.argmax(y_pred)])

        """
        Example Queries with CV + Random Forest - Tuned
        """
        print("\n--- Example Queries with CV + Random Forest - Tuned ---")
        print(new_extrinsic_review)
        chosen_model = self.loaded_models["Random Forest"][0]
        chosen_vect = self.loaded_models["Random Forest"][1]
        new_review = new_extrinsic_review
        x_query_dtm = chosen_vect.transform(new_review)
        y_pred = chosen_model.predict_proba(x_query_dtm)
        print(y_pred, labels[np.argmax(y_pred)])

def main():
    app = App()

if __name__ == "__main__":
    main()


"""
Some queries to test:

I love how all the colors in the Special K package make it more compelling to children. Everytime they see 
the box in the kitchen they believe they are threats and with that mentality they decide to eat this product!

I love the taste of my Special K

I hate Tony the Tiger
"""