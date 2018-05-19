import pandas as pd
from plotly.offline import plot
from plotly.graph_objs import Bar, Layout

bow = pd.read_csv('assets/bow.csv')

grouped = bow.groupby(['Category']).sum()
words_to_visualize = []
sentiments = ["positive","negative","neutral"]

for sentiment in sentiments:
  words = grouped.loc[sentiment,:]
  words.sort_values(inplace=True,ascending=False)
  for w in words.index[:7]:
    if w not in words_to_visualize:
      words_to_visualize.append(w)
            
plot_data = []
for sentiment in sentiments:
  plot_data.append(
    Bar(
      x = [w.split("_")[0] for w in words_to_visualize],
      y = [grouped.loc[sentiment, w] for w in words_to_visualize],
      name = sentiment
    ))
    
plot({"data":plot_data, "layout":Layout(title="Palavras mais comuns por sentimentos")})