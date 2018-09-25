from flask import request, jsonify, json
from . import app

# solve challenge
@app.route('/', methods=['POST'])
def solve_challenge():
    if request.headers['Content-Type'] == 'application/json':
        return (str)(request.json["challenge"])

