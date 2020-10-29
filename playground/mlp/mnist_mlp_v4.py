'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing import image

print('Loading model...')
model = keras.models.load_model('models')

source = Image.open('./images/2.png', 'r')
source = source.convert('L')
source = source.resize((28, 28))
source.show(command='fim')
print(source)

img_width, img_height = 28, 28
# test_image = image.load_img('./images/5.png', color_mode='grayscale', target_size=(img_width, img_height))
test_image = image.img_to_array(source)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image.reshape(1, img_width * img_height)
test_image = 255 - test_image
test_image = test_image.astype('float32')
test_image /= 255
result = model.predict(test_image, batch_size=1)
print(np.argmax(result))
