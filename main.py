import slack.slack_bot
import slack.slack_bot.app as app 
from textrank import get_summary

def process_text(text):
    info = get_summary(text)
    return "Keywords: " + ",".join(info[0]) + "\nSummary: too slow to show..." 

app.process_text = process_text

if __name__ == '__main__':  
    app.app.run(debug=True)

