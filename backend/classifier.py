import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import ImageFile, Image
from numpy import expand_dims
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
# from tensorflow.keras.Model import load_weights
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
ImageFile.LOAD_TRUNCATED_IMAGES = True

# model = InceptionV3(weights="imagenet", include_top=False)
PATH_TO_CLASSIFIER = "../research/models/weights_inception.hdf5"
classifier = load_model(PATH_TO_CLASSIFIER)

def get_prediction(image_as_bytes, model):
    raw_image = Image.open(image_as_bytes)
    converted_image = raw_image.convert("RGB").resize((299, 299), Image.NEAREST)

    image_tensor = image.img_to_array(converted_image)
    batched_tensor = expand_dims(image_tensor, axis=0)

    processed_tensor = preprocess_input(batched_tensor, mode="caffe")
    
    return model.predict(processed_tensor)


def classify_image(image_file):
    predicted_probabilities = get_prediction(image_file, classifier)

    prediction_tensor = decode_predictions(predicted_probabilities, top=1)
    
    return str(prediction_tensor[0][0][1])