# -*- coding: utf-8 -*-
"""
Gets the valuable data from the reviews data sets
Author: DanElias
Date: May 2021
"""
"""### Import python libraries"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

"""Use words related to Kellogg Company like products, brands, mascots"""

kelloggs_words = ['frosties', 'kellogg\'s', 'pop\start', 'froot\sloop',
                  'cheez-it', 'cheez\sit', 'corn\sflake', 'eggo', 'nutri-grain',
                  'nutri\sgrain', 'frosted\sflake', 'rice\skrispies',
                  'special\sk', 'cocoa\skrispies', 'tony\stiger',
                  'tony\sthe\stiger', 'toucan\ssam', 'pringles', 'all-bran',
                  'all\sbran', 'raisin-bran', 'raisin\sbran',
                  'morningstar\sfarms', 'corn\spop', 'krave', 'cheezit']

## (?:^|(?<=[\s:]))eggo(?=\s|$|s) each word can be at start of text, at the end, or in between spaces, never in between other chars
kelloggs_words = list(map(lambda word : '(?:^|(?<=[\s:]))' + word + '(?=\s|$|s)', kelloggs_words))             
regex = '|'.join(map(str, kelloggs_words))


"""# Tweets"""

tweets_df = pd.read_csv('../data/tweets.csv', encoding='UTF-8')
#tweets_df.head()
#tweets_df.shape

"""# Data preparation"""

tweets_df = tweets_df.dropna()
tweets_df['text'] = tweets_df['text'].str.lower() # Change to lower case
tweets_df['text'] = tweets_df['text'].str.replace('@\w*', '') # remove twitter usernames: @user
tweets_df['text'] = tweets_df['text'].str.replace('\B|[^a-z\'\-? ]', '') # remove special chars, minus '
#tweets_df.head()
#tweets_df.shape

"""### Only use tweets that talk about Kellogg's
I make this using a the of words that have to do with Kellogg's Company
"""

kelloggs_tweets_df = tweets_df[tweets_df["text"].str.contains(regex)]
#kelloggs_tweets_df.shape

"""Separate each text into sentences"""

sentences_twitter = np.array([])
for row in kelloggs_tweets_df["text"]:
  sentences_row = np.array(sent_tokenize(row))
  sentences_twitter = np.concatenate((sentences_twitter, sentences_row))
sentences_twitter.shape

"""Create Kellogg's Tweets DataFrame with the sentences"""

kelloggs_tweets_df = pd.DataFrame(sentences_twitter, columns=["text"])
#kelloggs_tweets_df.head()
#kelloggs_tweets_df.shape

"""# Amazon Reviews"""

amazon_df = pd.read_csv('../data/amazon_reviews.csv', encoding='UTF-8', usecols=["Summary", "Text"])
# amazon_df.head()
#amazon_df.shape

amazon_df = amazon_df.dropna()
amazon_df = amazon_df.rename(columns={"Text":"text"})
#amazon_df['text'] = amazon_df['text'].str.lower() # Change to lower case
amazon_df['text'] = amazon_df['text'].str.replace('@\w*', '') # remove twitter usernames: @user
amazon_df['text'] = amazon_df['text'].str.replace('\B|[^a-zA-Z\'.-? ]', '') # remove special chars, minus '
#amazon_df['Summary'] = amazon_df['Summary'].str.lower() # Change to lower case
amazon_df['Summary'] = amazon_df['Summary'].str.replace('@\w*', '') # remove twitter usernames: @user
amazon_df['Summary'] = amazon_df['Summary'].str.replace('\B|[^a-zA-Z\'.-? ]', '') # remove special chars, minus '
#amazon_df.head()
#amazon_df.shape

kelloggs_amazon_df = amazon_df[amazon_df["text"].str.contains(regex)]
#kelloggs_amazon_df.shape

"""Separate each text and review summary into sentences"""

sentences = np.array([])
for row in kelloggs_amazon_df["text"]:
  sentences_row = np.array(sent_tokenize(row))
  sentences = np.concatenate((sentences, sentences_row))
#sentences.shape

for row in kelloggs_amazon_df["Summary"]:
  sentences_row = np.array(sent_tokenize(row))
  sentences = np.concatenate((sentences, sentences_row))
#sentences.shape

"""Create the Kelloggs Amazon DataFrame with the sentences"""

kelloggs_amazon_df = pd.DataFrame(sentences, columns=["text"])
#kelloggs_amazon_df.head()

"""# Amazon Reviews 2"""

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

amazon2_df = pd.DataFrame()
amazon2_df["text"] = getDF('../data/reviews_Grocery_and_Gourmet_Food_5.json.gz')["reviewText"]
#amazon2_df.head()
#amazon2_df.shape

amazon2_df = amazon2_df.dropna()
#amazon2_df['text'] = amazon2_df['text'].str.lower() # Change to lower case
amazon2_df['text'] = amazon2_df['text'].str.replace('@\w*', '') # remove twitter usernames: @user
amazon2_df['text'] = amazon2_df['text'].str.replace('\B|[^a-zA-Z\'.-? ]', '') # remove special chars, minus '
#amazon2_df.head()
#amazon2_df.shape

kelloggs_amazon2_df = amazon2_df[amazon2_df["text"].str.contains(regex)]
#kelloggs_amazon2_df.shape

"""Separate each text into sentences"""

sentences_amazon = np.array([])
for row in kelloggs_amazon2_df["text"]:
  sentences_row = np.array(sent_tokenize(row))
  sentences_amazon = np.concatenate((sentences_amazon, sentences_row))
#sentences_amazon.shape

"""Create the Kellogg's Amazon 2 DataFrame with the sentences"""

kelloggs_amazon2_df = pd.DataFrame(sentences_amazon, columns=["text"])
kelloggs_amazon2_df.head()

#kelloggs_amazon2_df.shape

"""# Append Data from Twitter and Amazon
### And save it into a csv to be used by our Machine Learning checkpoint
"""

df =  pd.concat([kelloggs_tweets_df, kelloggs_amazon_df, kelloggs_amazon2_df])
#df.shape

df.to_csv("../data/kelloggs_reviews_test_unlabelled.csv", index=False)