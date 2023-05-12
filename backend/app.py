import os
from flask import Flask, render_template, request
from request import proxy_request
from classifier import classify_image

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
    
@app.route("/classify", methods-["POST"])
def classify():
    if request.files["image"]:
        image_file = request.files["image"]
        binary_classification = classify_image(image_file=image_file)
        print(f"Model Classification: {binary_classification}")
        return binary_classification