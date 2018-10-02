import slack.slack_bot.app
import textrank

def process_text(text):
  info = get_summary(text)
  return "Keywords: " + info[0].join(",") + "\nSummary: " + info[1]

if __name__ == '__main__':
  app.run(debug=True)

