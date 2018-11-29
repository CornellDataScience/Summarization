import slack.slack_bot
import slack.slack_bot.app as app 
from textrank import get_summary
from textrank2 import get_summary2
from kg import KG
import matplotlib.pyplot as plt
import networkx as nx

def summary(text, ts):
    keywords = get_summary(text)[0]
    summary = get_summary2(text) #change number of sentences?
    if len(text) > 1000:
        kg = KG()
        kg.make(text=text)
        nx.draw_networkx(kg.word_graph)
        plt.savefig("kg/" + str(ts)+".png", format="PNG")
        return [", ".join(keywords), summary, str(ts)]
    else:
        return [", ".join(keywords), summary, None]

app.summary = summary

if __name__ == '__main__':  
    app.app.run(debug=True)

