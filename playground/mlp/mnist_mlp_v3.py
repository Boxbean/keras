'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Loading model...')
model = keras.models.load_model('models')

test = x_test[100].reshape(28, 28)
plt.imshow(test)
test = x_test[100].reshape(1, 784)
result = model.predict(test)
print(y_test[100])
print(np.argmax(result))

img_width, img_height = 28, 28
test_image = image.load_img('./images/2.png', color_mode="grayscale", target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = test_image.astype('float32')
test_image = test_image.reshape(28, 28)
test_image = 255 - test_image
test_image /= 255
test_image = test_image.reshape(1, img_width * img_height)
result = model.predict(test_image, batch_size=1)
print(np.argmax(result))
