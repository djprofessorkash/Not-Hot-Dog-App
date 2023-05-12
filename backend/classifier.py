import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from numpy import expand_dims
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3

class ImageClassifier(object):
    def __init__(self):
        self.PATH_TO_CLASSIFIER = "../research/models/weights_inception.hdf5"
        self.classifier = load_model(self.PATH_TO_CLASSIFIER)

    def predict_probabilities(self, image_as_bytes):
        raw_image = Image.open(image_as_bytes)
        converted_image = raw_image.convert("RGB").resize((299, 299), Image.NEAREST)
        image_tensor = image.img_to_array(converted_image)
        batched_tensor = expand_dims(image_tensor, axis=0)
        processed_tensor = preprocess_input(batched_tensor, mode="caffe")
        return self.classifier.predict(processed_tensor)
    
    def classify_image(self, image_file):
        self.predicted_probabilities = self.predict_probabilities(image_file)
        self.binary_classification = decode_predictions(self.predicted_probabilities, top=1)[0][0][1]
