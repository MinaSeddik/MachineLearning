import matplotlib.pyplot as plt


import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers


if __name__ == '__main__':
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    input_dim = x_train.shape[1]
    encoding_dim = 32

    print("input_dim:", input_dim)


    compression_factor = float(input_dim) / encoding_dim
    print("Compression factor: %s" % compression_factor)

    model = Sequential()
    model.add(
        Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
    )
    model.add(
        Dense(input_dim, activation='sigmoid')
    )

    model.summary()

    input_img = Input(shape=(input_dim,))
    encoder_layer = model.layers[0]
    encoder = Model(input_img, encoder_layer(input_img))
    encoder.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=2)

