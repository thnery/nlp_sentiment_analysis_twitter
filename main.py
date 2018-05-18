#!/usr/bin/env python

import csv
import pandas as pd
import numpy as np
import string
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import sklearn.metrics as sm

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

def read_csv(path):
  return pd.read_csv(path, delimiter=",")

def write_csv(scores, accuracies, precisions, recalls, f1_scores, model):
  with open('assets/results_' + model + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['fold', 'score', 'acuracia', 'precisao', 'recall', 'f1_score'])
    for i in range(0, 10):
      k = i + 1
      score = scores[i]
      accuracy = accuracies[i]
      precision = precisions[i]
      recall = recalls[i]
      f1_score = f1_scores[i]
      writer.writerow([k, score, accuracy, precision, recall, f1_score])

def text_process(text):
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def main():
  data = read_csv('assets/train.csv')
  print(data.shape)
  print(data.head())

  kf = KFold(n_splits=10, shuffle=True, random_state=None)
  X, y = data['Tweet'], data['Category']

  # Transforms the text into a tokens matrix
  vectorizer = CountVectorizer(analyzer=text_process).fit(X)
  print(len(vectorizer.vocabulary_))
  X = vectorizer.transform(X)
  print('Shape of Sparse Matrix: ', X.shape)
  print('Amount of Non-Zero occurrences: ', X.nnz)

  # Encode labels
  le = preprocessing.LabelEncoder()
  le.fit(y)
  y = le.transform(y)
  print(le.classes_)

  scores = []
  accuracies = []
  precisions = []
  recalls = []
  f1_scores = []

  for train, test in kf.split(data):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    # Create the classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    score = clf.score(X_test, y_test)
    accuracy = sm.accuracy_score(y_test, y_pred)
    precision = sm.precision_score(y_test, y_pred, average="macro")
    recall = sm.recall_score(y_test, y_pred, average="macro")
    f1_score = sm.f1_score(y_test, y_pred, average="macro")

    # print(confusion_matrix(y_test, preds))
    print('\n')
    print("accuracy: %.4f precision: %.4f recall: %.4f f1-score: %.4f" % (accuracy, precision, recall, f1_score))
    print('\n')

    scores.append(score)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

  write_csv(scores, accuracies, precisions, recalls, f1_scores, 'MultinomialNB')



# call the main function
main()
