# -*- coding : utf-8 -*- 
import requests
import json
from . import bot
from flask import Flask, request, make_response, render_template, send_file

pyBot = bot.Bot()
slack = pyBot.client

app = Flask(__name__)

def summary(text, ts):
    return None

def _event_handler(event_type, slack_event):
    """
    A helper function that routes events from Slack to our Bot
    by event type and subtype.

    Parameters
    ----------
    event_type : str
        type of event recieved from Slack
    slack_event : dict
        JSON response from a Slack reaction event

    Returns
    ----------
    obj
        Response object with 200 - ok or 500 - No Event Handler error

    """
    team_id = slack_event["team_id"]
    # ================ Team Join Events =============== #
    # When the user first joins a team, the type of event will be team_join
    #if event_type == "team_join":
    #    user_id = slack_event["event"]["user"]["id"]
        # Send the onboarding message
    #    pyBot.onboarding_message(team_id, user_id)
    #    return make_response("Welcome Message Sent", 200,)

    # ============== Share Message Events ============= #
    # If the user has shared the onboarding message, the event type will be
    # message. We'll also need to check that this is a message that has been
    # shared by looking into the attachments for "is_shared".
    if event_type == "message":
        print(str(slack_event))
        user = slack_event["event"].get("user")
        message = slack_event["event"]
        if ((not "bot_id" in message) and "text" in message and len(message["text"].split()) > 50):
            ts = message.get("ts")
            update_ts = str(float(message.get("ts")) + 0.0000001)
            # update, fail authentification...
            # data = { 
            #    'token':"xoxb-9179452085-446961855171-u5xGDJOGl3BvR1nJGaBxF56m", 
            #    "channel":message.get("channel"), 
            #    "ts":ts,
            #    "as_user": True,
            #    "attachments":[ {                                                                    
            #        "fallback": "Required plain-text summary of the attachment.",    
            #        "color": "#36a64f",                                              
            #        "pretext": "Optional text that appears above the attachment block",
            #        "author_name": "Summarization team",                             
            #        "title": "app test",                                             
            #        "text": "Place holder for summarization",                        
            #        "ts": update_ts                                                         
            #   }]
            #}
            #r = requests.post("https://slack.com/api/chat.update", data)
            #print(r.text)
            s = summary(message.get("text"), ts)
            data = {
                'token': "xoxb-9179452085-446961855171-u5xGDJOGl3BvR1nJGaBxF56m",
                "channel": message.get("channel"),
                "text": "It is better to have a summary...",
                "as_user": True,
                "thread_ts": ts,
                "attachments": [
                    {
                        "title": "Keywords",
                        "text": s[0]
                    },
                    {
                        "title": "Key sentences",
                        "text": s[1]
                    },
                    {
                        "title": "Graph",
                        "image_url": "http://128.84.48.178/get_image?ts=" + s[2]
                    },
                    {
                        "fallback": "Do you like the summary?",
                        "title": "Do you like the summary?",
                        "callback_id": "feedback",
                        "color": "#3AA3E3",
                        "attachment_type": "default",
                        "actions": [
                            {
                                "name": "yes",
                                "text": "yes",
                                "type": "button",
                                "value": "good"
                            },
                            {
                                "name": "no",
                                "text": "No",
                                "type": "button",
                                "value": "bad"
                            }
                        ]
                    }
                ]
            }
            r = requests.post("https://slack.com/api/chat.postMessage", data)
            return make_response("respond", 200,)

    if event_type == "interactive_message":
        print(slack_event["actions"])
        print(slack_event["original_message"])

    # ============= Reaction Added Events ============= #
    # If the user has added an emoji reaction to the onboarding message
    #elif event_type == "reaction_added":
    #    user_id = slack_event["event"]["user"]
        # Update the onboarding message
    #    pyBot.update_emoji(team_id, user_id)
    #    return make_response("Welcome message updates with reactji", 200,)

    # =============== Pin Added Events ================ #
    # If the user has added an emoji reaction to the onboarding message
    #elif event_type == "pin_added":
    #    user_id = slack_event["event"]["user"]
        # Update the onboarding message
    #    pyBot.update_pin(team_id, user_id)
    #    return make_response("Welcome message updates with pin", 200,)

    # ============= Event Type Not Found! ============= #
    # If the event_type does not have a handler
    message = "You have not added an event handler for the %s" % event_type
    # Return a helpful error message
    return make_response(message, 200, {"X-Slack-No-Retry": 1})


@app.route("/install", methods=["GET"])
def pre_install():
    """This route renders the installation page with 'Add to Slack' button."""
    # Since we've set the client ID and scope on our Bot object, we can change
    # them more easily while we're developing our app.
    client_id = pyBot.oauth["client_id"]
    scope = pyBot.oauth["scope"]
    # Our template is using the Jinja templating language to dynamically pass
    # our client id and scope
    return render_template("install.html", client_id=client_id, scope=scope)


@app.route("/thanks", methods=["GET", "POST"])
def thanks():
    """
    This route is called by Slack after the user installs our app. It will
    exchange the temporary authorization code Slack sends for an OAuth token
    which we'll save on the bot object to use later.
    To let the user know what's happened it will also render a thank you page.
    """
    # Let's grab that temporary authorization code Slack's sent us from
    # the request's parameters.
    code_arg = request.args.get('code')
    # The bot's auth method to handles exchanging the code for an OAuth token
    pyBot.auth(code_arg)
    return render_template("thanks.html")


@app.route("/listening", methods=["GET", "POST"])
def hears():
    """
    This route listens for incoming events from Slack and uses the event
    handler helper function to route events to our Bot.
    """
    slack_event = json.loads(request.data)

    # ============= Slack URL Verification ============ #
    # In order to verify the url of our endpoint, Slack will send a challenge
    # token in a request and check for this token in the response our endpoint
    # sends back.
    #       For more info: https://api.slack.com/events/url_verification
    
    if "challenge" in slack_event:
        return make_response(slack_event["challenge"], 200, {"content_type":
                                                             "application/json"
                                                             })

    # ============ Slack Token Verification =========== #
    # We can verify the request is coming from Slack by checking that the
    # verification token in the request matches our app's settings
    if pyBot.verification != slack_event.get("token"):
        message = "Invalid Slack verification token: %s \npyBot has: \
                   %s\n\n" % (slack_event["token"], pyBot.verification)
        # By adding "X-Slack-No-Retry" : 1 to our response headers, we turn off
        # Slack's automatic retries during development.
        make_response(message, 403, {"X-Slack-No-Retry": 1})

    # ====== Process Incoming Events from Slack ======= #
    # If the incoming request is an Event we've subcribed to
    if "event" in slack_event:
        event_type = slack_event["event"]["type"]
        # Then handle the event by event_type and have your bot respond
        return _event_handler(event_type, slack_event)
    # If our bot hears things that are not events we've subscribed to,
    # send a quirky but helpful error response
    return make_response("[NO EVENT IN SLACK REQUEST] These are not the droids\
                         you're looking for.", 404, {"X-Slack-No-Retry": 1})


@app.route('/get_image')
def get_image():
    # return send_file(request.args.get('ts') + ".png", mimetype='image/gif')
    print(request.args.get('ts'))
    return send_file("placeholder.png", mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)


