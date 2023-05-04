"""
AUTHOR: Aakash 'Kash' Sudhakar
INTENT: The objective of this file is to analyze, process, and classify images
        scraped from the ImageNet dataset in order to construct a finetuned model 
        that can be used for hot dog classification.
NOTE:   Due to constraints in accessing the ImageNet dataset, this file is designed
        to handle a placeholder dataset accessed from Kaggle. This dataset can be 
        downloaded from https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog.
"""

# print("Loading libraries...")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from efficientnet.keras import EfficientNetB0
# print("Libraries loaded.")


class ClassificationEngine(object):
    def __init__(self, model):
        self.model = model
        self.configure()
        
    def configure(self):
        pass

    def run(self, model):
        pass


if __name__ == "__main__":
    engines = list()
    for model in [VGG16, ResNet50, InceptionV3, EfficientNetB0]:
        engine = ClassificationEngine()
        engine.run(model)
        engines.append(engine)