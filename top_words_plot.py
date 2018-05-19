import pandas as pd
import nltk.corpus as nc
import ast
from collections import Counter
from plotly.offline import plot
from plotly.graph_objs import Bar, Layout

data = pd.read_csv('assets/processed_data2.csv')
words = Counter()

for idx in data.index:
  words.update(ast.literal_eval(data.loc[idx, 'Tweet']))

for idx, stop_word in enumerate(nc.stopwords.words("english")):
  del words[stop_word]

word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common()],
          "occurrences": [v for k, v in words.most_common()]},
          columns=["word", "occurrences"])

words = list(word_df.loc[0:10,"word"])
words.reverse()
occ = list(word_df.loc[0:10,"occurrences"])
occ.reverse()

dist = [
  Bar(
    x = occ,
    y = words,
    orientation = "h"
)]
  
plot({"data":dist, "layout":Layout(title="Palavras mais frequentes na lista")})