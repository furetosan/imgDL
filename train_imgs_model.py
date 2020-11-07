#!/usr/bin/env python
# coding: utf-8

# **I am merely following their tutorial, even if modifying it.**

# ## Import TensorFlow and other libraries

# import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import imghdr
import argparse
import datetime as dt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


parser = argparse.ArgumentParser(description='Train Image Classifier.')
parser.add_argument("model") # , description='filename for model, possibly with .h5 extension')
args = parser.parse_args()
model_file_name = args.model

# ## Download and explore the dataset

# This __tutorial remake__ uses a dataset of about 3,700 photos of flowers. The dataset contains 5 sub-directories, one per class:
# 
# ```
# imgs/
#     full/*.jpg
#     game/*.jpg
#     meme/*,jpg
#     photo/*.jpg
#     poster/*.jpg
#     icon/*.jpg
#     logo/*.jpg
#     *.jpg
# ```

data_dir = pathlib.Path('imgs')
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Total: "+str(image_count))

game = list(data_dir.glob('game/*'))
meme = list(data_dir.glob('meme/*'))
photo = list(data_dir.glob('photo/*'))
poster = list(data_dir.glob('poster/*'))
icon = list(data_dir.glob('icon/*'))
logo = list(data_dir.glob('logo/*'))

batch_size = 32
img_height = 240
img_width = 320

# new_model = tf.keras.models.load_model(model_file_name)
# new_model.summary()

#data_augmentation = keras.Sequential(
#[
#   layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                input_shape=(img_height, 
#                                                             img_width,
#                                                             3)),
#    layers.experimental.preprocessing.RandomRotation(0.1),
#    layers.experimental.preprocessing.RandomZoom(0.1),
#   ]
# )

num_classes = 7
model = Sequential([
  # data_augmentation,
  # layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("[t] Started training: "+str(dt.datetime.now()))
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
print("[t] Finished training: "+str(dt.datetime.now()))

model.summary()
model.save(model_file_name) 

print('Model saved as: '+str(model_file_name))
