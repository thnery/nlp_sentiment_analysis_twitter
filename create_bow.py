import pandas as pd
import nltk.corpus as nc
import ast
from collections import Counter

def create_wordlist(data):
  words = Counter()

  for idx in data.index:
    words.update(ast.literal_eval(data.loc[idx, 'Tweet']))
  
  for idx, stop_word in enumerate(nc.stopwords.words("english")):
    del words[stop_word]
    
  return [k for k, v in words.most_common() if 3 < v < 500]

def create_bow(data, wordlist):
  rows = []
  wordlist = create_wordlist(data)
  columns = ['Category'] + list(map(lambda w: w + '_bow', wordlist))
  
  for idx in data.index:
    current_row = []

    current_label = data.loc[idx, 'Category']
    current_row.append(current_label)
  
    tokens = set(ast.literal_eval(data.loc[idx, 'Tweet']))
    for _, word in enumerate(wordlist):
      current_row.append(1 if word in tokens else 0)
    
    rows.append(current_row)
  
  return pd.DataFrame(rows, columns=columns)

data = pd.read_csv('assets/processed_data2.csv')

wordlist = create_wordlist(data)
bow = create_bow(data, wordlist)
bow.to_csv('assets/bow.csv', index=False)