import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from time import time
from sklearn.model_selection import cross_val_score
import random
seed = 666
random.seed(seed)

def cv(classifier, X_train, y_train):
  log("===============================================")
  classifier_name = str(type(classifier).__name__)
  now = time()
  log("Crossvalidating " + classifier_name + "...")
  accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
  log("Crosvalidation completed in {0}s".format(time() - now))
  log("Accuracy: " + str(accuracy[0]))
  log("Average accuracy: " + str(np.array(accuracy[0]).mean()))
  log("===============================================")
  return accuracy

def log(x):
  print(x)
            
bow = pd.read_csv('assets/bow.csv')

nb_acc = cv(BernoulliNB(), bow.iloc[:,1:], bow.iloc[:,0])
  