# lstm autoencoder predict sequence
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#   https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras/


if __name__ == '__main__':
    X = list()
    Y = list()
    X = [x + 1 for x in range(20)]
    Y = [y * 15 for y in X]

    print(X)
    print(Y)

    # The input to LSTM layer should be in 3D shape i.e. (samples, time-steps, features)
    X = array(X).reshape(20, 1, 1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit(X, Y, epochs=3000, validation_split=0.2, batch_size=5)

    test_input = array([30, 100, 25])
    test_input = test_input.reshape((3, 1, 1))
    test_output = model.predict(test_input, verbose=0)
    print("test_output.shape", test_output.shape)
    print("test_output = ", test_output)

