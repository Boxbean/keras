'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import numpy as np
from PIL import Image,ImageOps
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing import image

import os

print('Loading model...')
model = keras.models.load_model('models')
def predict(path):
    source = Image.open(path, 'r')
    source = source.convert('L')
    source = source.resize((28, 28))
    source = ImageOps.invert(source) # invert
    # source.show(command='fim')

    img_width, img_height = 28, 28
    test_image = np.array(source)
    test_image = test_image.astype('float32')
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image.reshape(1, img_height* img_width)
    result = model.predict(test_image)
    print(np.argmax(result))

images = os.listdir("./images")
print(images)
for image in images:
    if image == ".DS_Store":
        continue
    path = "./images/" + str(image)
    predict(path)