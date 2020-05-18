from __future__ import division, print_function
from flask import jsonify, make_response
# coding=utf-8
import sys
import os


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



if __name__ == '__main__':
    #app.jinja_env.cache = {}
    #app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

