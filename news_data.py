'''
Cornell Data Science Fall 2018
Text summarization group: Wes Gurnee, Qian Huang, Jane Zhang

Script for placing articles into text files from All the News kaggle dataset.

There are 3 varibles/expressions you have to modify which are all indicated
by comments.'''

import re
import pandas as pd
#Change to where your data lives
CSV_FILE_PATH = 'Data/articles2.csv'
#Where you want the text files to be dumped, make sure this directory exists
SAVE_PATH = 'Data/trump_russia/'

text = pd.read_csv(CSV_FILE_PATH)

article_ids = []
for ix, row in text.iterrows():
    try:
        #This is where you create your subsetting conditions
        if 'Trump' in row.title  \
        and 'Russia' in row.title \
        and row.publication == 'Atlantic':
            article_ids.append(ix)
    except:
        continue

for ix, row in text.iloc[article_ids].iterrows():
    save_name = str(ix) + '-' + re.sub(r'\W+', '', row.title.replace(' ', '_')[:50]) + '.txt'
    with open(SAVE_PATH + save_name, "w", encoding="utf-8") as text_file:
        text_file.write(row.content)
