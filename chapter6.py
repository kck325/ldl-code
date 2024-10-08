# Towards DK: Frameworks and network tweaks

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

EPOCHS = 20
BATCH_SIZE = 1

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

train_labels = to_categorical(train_labels, num_classes=10)
test_lables = to_categorical(test_labels, num_classes=10)

initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # For converting 28x28 images into 784x1 vectors - 1D
    keras.layers.Dense(units=25, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Dense(units=10, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
])

opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)