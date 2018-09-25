from flask import Flask, render_template, jsonify, make_response
import os

# Configure Flask app
app = Flask(__name__, static_url_path='/static')

import controller


# HTTP error handling
@app.errorhandler(404)
def not_found(error):
  return render_template('404.html'), 404
