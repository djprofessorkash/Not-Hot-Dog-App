import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import ImageFile, Image
from numpy import expand_dims
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = InceptionV3(weights='imagenet', include_top=False)

def get_prediction(image_as_bytes, model):
    pass

def classify_image(image_file):
    pass