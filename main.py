#!/usr/bin/env python

import pandas as pd
import numpy as np
import string
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing

def read_csv(path):
  return pd.read_csv(path, delimiter=",")

def text_process(text):
  '''
  Takes in a string of text, then performs the following:
  1. Remove all punctuation
  2. Remove all stopwords
  3. Return the cleaned text as a list of words
  '''
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

data = read_csv('assets/train.csv')
print(data.shape)
print(data.head())

X, y = data['Tweet'], data['Category']
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
print(len(bow_transformer.vocabulary_))

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
print(y)

X = bow_transformer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

nb = MultinomialNB()
nb.fit(X_train, y_train)

preds = nb.predict(X_test)

print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
