import os
from flask import Flask, render_template, request
from request import proxy_request
from classifier import ImageClassifier

MODE = os.getenv("FLASK_ENV")
DEV_SERVER_URL = "http://localhost:3000/"

app = Flask(__name__)

# Ignore static folder in development mode
if MODE == "development":
    app = Flask(__name__, static_folder=None)

@app.route("/")
@app.route("/<path:path>")
def index(path=""):
    if MODE == "development":
        return proxy_request(DEV_SERVER_URL, path)
    else:
        return render_template("index.html")
    
@app.route("/classify", methods=["POST"])
def classify():
    if (request.files["image"]):
        classifier, image_file = ImageClassifier(), request.files["image"]
        classifier.classify_image(image_file=image_file)
        print(f"Model Classification: {str(classifier.binary_classification)}")
        return classifier.binary_classification