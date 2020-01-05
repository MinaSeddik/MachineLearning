from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


from keras.utils import to_categorical

from mnist_reader import load_mnist_dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    img_size = 28

    # load MNIST dataset
    logger.debug('loading MNIST dataset')
    x_train, y_train_label, x_test, y_test_label = load_mnist_dataset()

    # we need to convert the label to hot encoded format
    y_train = to_categorical(y_train_label)
    y_test = to_categorical(y_test_label)

    logger.debug('x_train shape = %s', x_train.shape)
    logger.debug('y_train shape = %s', y_train.shape)

    logger.debug('Before Normalizing, First Image = \n%s', x_train[0])

    # prepare and normalize the training and test sets
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    logger.debug('After Normalizing, First Image = \n%s', x_train[0])
    logger.debug('First Label = %s', y_train_label[0])
    logger.debug('First Hot encoded Label = %s', y_train[0])

    model = tf.keras.Sequential()

    model.add(layers.Dense(64, input_shape=(img_size * img_size,), activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary(print_fn=lambda line: logger.info(line))

    # compile the model
    learning_rate = 0.001

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])



    # train the model
    model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=100, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info("Test accuracy: %.2f%%" % (test_acc * 100))




