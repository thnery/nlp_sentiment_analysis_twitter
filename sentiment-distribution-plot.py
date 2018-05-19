import pandas as pd
from plotly.offline import plot
from plotly.graph_objs import Bar, Layout

data = pd.read_csv('assets/data.csv')

neg = len(data[data["Category"] == "negative"])
pos = len(data[data["Category"] == "positive"])
neu = len(data[data["Category"] == "neutral"])

dist = [
  Bar(
    x = ["Negativo","Neutro","Positivo"],
    y = [neg, neu, pos],
)]
    
plot({"data":dist, "layout":Layout(title="Distribuição de Sentimento da Base")})