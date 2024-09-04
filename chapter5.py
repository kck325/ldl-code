# Fully connected neural networks applied to regression

# Regression problem vs classification problem
# Regression problem: predicting a continuous output value
# Classification problem: predicting a discrete output value

# Loss function:
# Cross entropy loass: Logistic outout (binary classification)
# categorical cross entropy loss: softmax output (multiclass classification)
# mean squared error: linear output (regression)
 # what is logit? - Inverser of logistic function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import logging
tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 500
BATCH_SIZE = 16

boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()
x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0)
x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
predictions = model.predict(x_test)
for i in range(0, 4):
    print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")


# Regularization techniques
# L1 regularization: adds a penalty term to the loss function that encourages the weights to be sparse
# L2 regularization: cross entropy loss + (lambda * sum of all weights^2)
# Dropout: removes some neurons randomly to prevent coadpating