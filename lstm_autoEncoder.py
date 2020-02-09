# lstm autoencoder predict sequence
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.utils import plot_model
from numpy import array

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# https://machinelearningmastery.com/lstm-autoencoders/

if __name__ == '__main__':
    # define input sequence
    seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # reshape input into [samples, timesteps, features]
    n_in = len(seq_in)
    seq_in = seq_in.reshape((1, n_in, 1))
    print(seq_in)

    # prepare output sequence
    seq_out = seq_in[:, 1:, :]
    print(seq_out)
    n_out = n_in - 1

    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_out))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.summary(print_fn=lambda line: logger.info(line))

    model.compile(optimizer='adam', loss='mse')
    # plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')

    # fit model
    model.fit(seq_in, seq_out, epochs=100, verbose=0)

    # demonstrate prediction
    yhat = model.predict(seq_in, verbose=0)
    print(yhat[0, :, 0])
