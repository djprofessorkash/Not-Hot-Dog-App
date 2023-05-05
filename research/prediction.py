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
import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


class DistributionEngine(object):
	""" Standalone class for distributing loaded images into training, testing, and validation sets. """
	def __init__(self, dirpath="research/data", segments=(0.8, 0.1, 0.1)):
		self.DIRPATH, self.train_val_test_split = dirpath, segments
		self.POSCLASS, self.NEGCLASS = "hotdog", "nothotdog"
		self.SUBDIRTRAIN, self.SUBDIRVAL, self.SUBDIRTEST = "training", "validation", "testing"

	def distribute(self):
		for label in [self.POSCLASS, self.NEGCLASS]:
			for subdirectory in [self.SUBDIRTRAIN, self.SUBDIRVAL, self.SUBDIRTEST]:
				os.makedirs(f"{self.DIRPATH}/{subdirectory}/{label}/", exist_ok=True)
			sources = f"{self.DIRPATH}/{label}"
			targets = os.listdir(sources); np.random.shuffle(targets)
			splits = np.split(np.array(targets),
		     				  [int(len(targets) * self.train_val_test_split[0]),
	      					   int(len(targets) * (self.train_val_test_split[0] + self.train_val_test_split[1]))])
			training_group = [f"{sources}/{filepath}" for filepath in splits[0].tolist()]
			validation_group = [f"{sources}/{filepath}" for filepath in splits[1].tolist()]
			testing_group = [f"{sources}/{filepath}" for filepath in splits[2].tolist()]
			for filename in training_group:
				shutil.copy(filename, f"{self.DIRPATH}/{self.SUBDIRTRAIN}/{label}")
			for filename in validation_group:
				shutil.copy(filename, f"{self.DIRPATH}/{self.SUBDIRVAL}/{label}")
			for filename in testing_group:
				shutil.copy(filename, f"{self.DIRPATH}/{self.SUBDIRTEST}/{label}")


class ClassificationEngine(object):
    def __init__(self, model):
        self.model = model
        self.configure()

    def configure(self):
        pass

    def run(self, model):
        pass


if __name__ == "__main__":
	# Redistribute images into training, validation, and testing sets
	# redistributor = DistributionEngine()
	# redistributor.distribute()

	# Create pretrained engines for comparative assessments
    engines = list()
    for model in [VGG16, ResNet50, InceptionV3, EfficientNetB0]:
        engine = ClassificationEngine()
        engine.run(model)
        engines.append(engine)