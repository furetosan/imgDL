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
import subprocess

from IPython import get_ipython
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


parser = argparse.ArgumentParser(description='Classify Images.')
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

##
##

new_model = tf.keras.models.load_model(model_file_name)
class_names = ['full', 'game', 'icon', 'logo', 'meme', 'photo', 'poster']

print(class_names)

# ## Predict on new data

data_dir = pathlib.Path('imgs')

def get_predictions():
    new_imgs = list(data_dir.glob('*.jpg')) #, data_dir.glob('full/*.jpg')
    previsoes = list()

    for ni in new_imgs:
        img = keras.preprocessing.image.load_img(
            ni, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        get_ipython().system('tycat '+str(ni))
        # !tycat {"imgs/"+str(ni)}
        # subprocess.run( [ "tycat", "ni" ] )

        print(
            "The image {} most likely belongs to {} with a {:.2f} percent confidence."
            .format(ni,class_names[np.argmax(score)], 100 * np.max(score))
        )
        previsoes.append([ni,class_names[np.argmax(score)], 100 * np.max(score)])
        mexe = input("Move (y/n)?")
        if mexe == 'y':
            os.rename(ni, "imgs/" + class_names[np.argmax(score)]+str(ni)[4:] )
    return previsoes


get_predictions()
