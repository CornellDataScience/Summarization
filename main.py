import slack.slack_bot
import slack.slack_bot.app as app 
from textrank import get_summary
from textrank2 import get_summary2

def summary(text, ts):
    keywords = get_summary(text)[0]
    summary = get_summary2(text) #change number of sentences?
    return [", ".join(keywords), summary, str(ts)]

app.summary = summary

if __name__ == '__main__':  
    app.app.run(debug=True)

