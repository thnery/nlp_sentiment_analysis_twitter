import pandas as pd
import re

data = pd.read_csv('assets/data.csv')

special_characters = [",", ":", "\"", "=", "&", ";", "%", "$", "@", "%", "^", 
                      "*", "(", ")", "{", "}", "[", "]", "|", "/", "\\", ">", 
                      "<", "-", "!", "?", ".", "'", "--", "---", "#"
]

regexes = [
    re.compile(r"http.?://[^\s]+[\s]?"), # Select URLs
    re.compile(r"@[^\s]+[\s]?"),         # Select mentions
    re.compile(r"\s?[0-9]+\.?[0-9]*")    # Select numbers
]

for regex in regexes:
  data.loc[:, 'Tweet'].replace(regex, '', inplace=True)

# Remove special characters
for remove in map(lambda r: re.compile(re.escape(r)), special_characters):
  data.loc[:, 'Tweet'].replace(remove, '', inplace=True)

# Remove tweets with Not Available text
data = data[data['Tweet'] != 'Not Available']

data.to_csv('assets/processed_data.csv')