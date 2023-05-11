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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.efficientnet import EfficientNetB0
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
	def __init__(self, dirpath="research/data"):
		self.DIRPATH, self.POSCLASS, self.NEGCLASS = dirpath, "hotdog", "nothotdog"
		self.SUBDIRTRAIN, self.SUBDIRVAL, self.SUBDIRTEST = "training", "validation", "testing"
		self.models = {
			"inception": {"classifier": InceptionV3, "loss": 0, "accuracy": 0},
			"vgg16": {"classifier": VGG16, "loss": 0, "accuracy": 0},
            "resnet50": {"classifier": ResNet50, "loss": 0, "accuracy": 0},
            "efficientnet": {"classifier": EfficientNetB0, "loss": 0, "accuracy": 0}
		}
		self.BATCHSIZE, self.EPOCHS = 32, 50
	
	def load(self):
		self.generator = ImageDataGenerator(rescale=1./255)
		self.training_generator = self.generator.flow_from_directory(
			directory=f"{self.DIRPATH}/{self.SUBDIRTRAIN}",
			target_size=(299, 299),
			batch_size=self.BATCHSIZE,
			class_mode=None,
			shuffle=False)
		self.validation_generator = self.generator.flow_from_directory(
            directory=f"{self.DIRPATH}/{self.SUBDIRVAL}",
            target_size=(299, 299),
            batch_size=self.BATCHSIZE,
            class_mode=None,
			shuffle=False)
		self.testing_generator = self.generator.flow_from_directory(
            directory=f"{self.DIRPATH}/{self.SUBDIRTEST}",
            target_size=(299, 299),
            batch_size=self.BATCHSIZE,
            class_mode=None,
			shuffle=False)
		self.CLASSES = [outcome for outcome in self.testing_generator.class_indices.keys()]

	def configure(self, model):
		self.pretrained_model = self.models[model]["classifier"](weights="imagenet", include_top=False, input_shape=(299, 299, 3))

		bottleneck_training_features = self.pretrained_model.predict(self.training_generator, 
							      									 self.training_generator.samples / self.BATCHSIZE, 
																	 verbose=1)
		np.savez(f"{self.DIRPATH}/{self.SUBDIRTRAIN}/training_features:{model}", features=bottleneck_training_features)
		bottleneck_validation_features = self.pretrained_model.predict(self.validation_generator, 
                                                                   	   self.validation_generator.samples / self.BATCHSIZE, 
                                                                   	   verbose=1)
		np.savez(f"{self.DIRPATH}/{self.SUBDIRVAL}/validation_features:{model}", features=bottleneck_validation_features)
		bottleneck_testing_features = self.pretrained_model.predict(self.testing_generator, 
                                                                   	self.testing_generator.samples / self.BATCHSIZE, 
                                                                   	verbose=1)
		np.savez(f"{self.DIRPATH}/{self.SUBDIRTEST}/testing_features:{model}", features=bottleneck_testing_features)

		self.X_training = np.load(f"{self.DIRPATH}/{self.SUBDIRTRAIN}/training_features:{model}.npz")["features"]
		self.X_validation = np.load(f"{self.DIRPATH}/{self.SUBDIRVAL}/validation_features:{model}.npz")["features"]
		self.X_testing = np.load(f"{self.DIRPATH}/{self.SUBDIRTEST}/testing_features:{model}.npz")["features"]
		self.y_training = np.array(([0] * int(self.training_generator.samples / 2)) + ([1] * int(self.training_generator.samples / 2)))
		self.y_validation = np.array(([0] * int(self.validation_generator.samples / 2)) + ([1] * int(self.validation_generator.samples / 2)))
		self.y_testing = np.array(([0] * int(self.testing_generator.samples / 2)) + ([1] * int(self.testing_generator.samples / 2)))
	
	def run(self, model):
		self.classifier = Sequential()
		self.classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=self.X_training.shape[1:], padding="same"))
		self.classifier.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
		self.classifier.add(MaxPooling2D(pool_size=(3, 3)))
		self.classifier.add(Dropout(0.25))
		self.classifier.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
		self.classifier.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
		self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
		self.classifier.add(Dropout(0.5))
		self.classifier.add(Flatten())
		self.classifier.add(Dense(512, activation="relu"))
		self.classifier.add(Dropout(0.6))
		self.classifier.add(Dense(256, activation="relu"))
		self.classifier.add(Dropout(0.5))
		self.classifier.add(Dense(1, activation="sigmoid"))
		checkpointer = ModelCheckpoint(filepath=f"research/models/weights_{model}.hdf5", verbose=1, save_best_only=True)
		stopper = EarlyStopping(monitor="val_loss", patience=12, verbose=1, mode="auto")
		lrreducer = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.00001)
		print(self.classifier.summary())
		self.classifier.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["binary_accuracy"])
		self.history = self.classifier.fit(self.X_training, 
				     					   self.y_training, 
										   epochs=50, 
										   batch_size=self.BATCHSIZE,
										   validation_data=(self.X_validation, self.y_validation), 
										   callbacks=[checkpointer, stopper, lrreducer],
										   shuffle=True,
										   verbose=2)
		self.classifier.load_weights(f"research/models/weights_{model}.hdf5")
		self.y_predictions = self.classifier.predict(self.X_testing, batch_size=self.BATCHSIZE, verbose=1)
		self.models[model]["loss"], self.models[model]["accuracy"] = self.classifier.evaluate(self.X_testing,
																							  self.y_testing, 
																							  verbose=1, 
																							  batch_size=self.BATCHSIZE)


if __name__ == "__main__":
	# Redistribute images into training, validation, and testing sets
	# redistributor = DistributionEngine()
	# redistributor.distribute()

	# Create predictive engine for comparative assessments
	classifier = ClassificationEngine()
	classifier.load()
	for model in ["inception", "vgg16", "resnet50", "efficientnet"]:
		classifier.configure(model)
		classifier.run(model)
	for model in ["inception", "vgg16", "resnet50", "efficientnet"]:
		print(f">> {model}: {classifier.models[model]['loss']:.4f}, {classifier.models[model]['accuracy']:.4f}")
