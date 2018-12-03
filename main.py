import slack.slack_bot
import slack.slack_bot.app as app
from textrank import get_summary
from textrank2 import get_summary2
from kg import KG
import matplotlib.pyplot as plt
import networkx as nx
import os
import nltk

def summary(text, ts):
    keywords = get_summary(text)[0]
    num = min(len(text.split())/ 100, 3)
    summary = get_summary2(text, numSentences= num)
    if len(text) > 1000:
        kg = KG()
        kg.make(edge_words = True, text=text)
        plt.savefig("kg/" + str(ts)+".png", format="PNG")
        return [", ".join(keywords), summary, str(ts)]
    else:
        return [", ".join(keywords), summary, "N/A"]

app.summary = summary

if __name__ == '__main__':
    app.app.run(debug=True)

