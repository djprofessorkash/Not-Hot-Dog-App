import os
from flask import Flask, render_template
from request import proxy_request

MODE = os.getenv("FLASK_ENV")
DEV_SERVER_URL = "http://localhost:3000"

app = Flask(__name__)

# Ignore static folder in development mode
if MODE == "development":
    app = Flask(__name__, static_Folder=None)

@app.route("/")
@app.route("/<path:path>")
def index(path=""):
    if MODE == "development":
        return proxy_request(DEV_SERVER_URL, path)
    else:
        return render_template("index.html")