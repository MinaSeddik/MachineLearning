import logging

import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from mnist_reader import load_mnist_dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    img_size = 28

    # load MNIST dataset
    logger.debug('loading MNIST dataset')
    x_train, y_train_label, x_test, y_test_label = load_mnist_dataset()

    # reshape the x_train and x_test to fit into the convolutional NN model
    x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
    x_test = x_test.reshape(x_test.shape[0], img_size, img_size, 1)

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

    # create model
    model = Sequential()

    # add model layers
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary(print_fn=lambda line: logger.info(line))

    # compile the model
    learning_rate = 0.001

    logger.info('Compile the model with Learning rate = %f', learning_rate)
    model.compile(Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=100, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info("Test accuracy: %.2f%%" % (test_acc * 100))

    # Save the model to disk.
    model_file_path = 'saved_models/mnist_keras_cnn.h5'
    model.save_weights(model_file_path)

    # Load the model from disk later using:
    # model.load_weights(model_file_path)

    # make class prediction
    # rounded_predictions = model.predict_classes(x_test)

    # Predict on the test images.
    predictions = model.predict(x_test)
    # HINT: the prediction is ratio of all classes per test row

    logger.debug('x_test shape = %s', x_test.shape)
    logger.debug('predictions shape = %s', predictions.shape)

    # get the max class for each
    y_predict = np.argmax(predictions, axis=1)

    logger.debug("Display the First 10 test Image's predictions:")
    logger.debug('\tLabel\t\tPredicted Label:')
    for i in range(0, 10):
        logger.debug('\t%d\t\t%d', y_test_label[i], y_predict[i])
