import pandas as pd
import nltk
nltk.download('punkt')

data = pd.read_csv('assets/processed_data.csv')

def tokenize(row):
  row['Tweet'] = nltk.word_tokenize(row['Tweet'])
  return row
          
def stem(row):
  row['Tweet'] = list(map(lambda str: nltk.PorterStemmer().stem(str.lower()), row['Tweet']))
  return row

data = data.apply(tokenize, axis=1)
data = data.apply(stem, axis=1)

data.to_csv('assets/processed_data2.csv')