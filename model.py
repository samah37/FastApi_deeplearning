import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.imagenet_utils import (decode_predictions, preprocess_input)
from tensorflow.keras.preprocessing.image import img_to_array

def load_model():
    model =  ResNet50(weights= "imagenet")
    print("Model loaded")
    return model

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image
def predict(image, model):
    results = decode_predictions(model.predict(image), 2)[0]
    response =  [
        {"class":result[1], "score": float(round(result[2],3))} for result in results
    ]
    return response
