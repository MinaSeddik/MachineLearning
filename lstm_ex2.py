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
# https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714
# https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e


if __name__ == '__main__':
    nums = 25

    X1 = list()
    X2 = list()
    X = list()
    Y = list()

    X1 = [(x + 1) * 2 for x in range(25)]
    X2 = [(x + 1) * 3 for x in range(25)]
    Y = [x1 * x2 for x1, x2 in zip(X1, X2)]

    print(X1)
    print(X2)
    print(Y)

    X = np.column_stack((X1, X2))
    print(X)



