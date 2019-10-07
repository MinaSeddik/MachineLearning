import csv
import logging
import os

import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
# from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def read_iris_data():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    iris_data_file = os.path.join(root_dir, 'data', 'Iris', 'Iris.csv')

    logger.debug('reading Iris data from: %s', iris_data_file)

    with open(iris_data_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # convert it to numpy array
    data = np.array(data)

    # skipping first row - the header
    logger.debug('data shape: %s', data.shape)
    data = data[1:, :]
    logger.debug('data shape (After skipping the header) : %s', data.shape)

    # skipping first column - the Id
    data = data[:, 1:]
    logger.debug('data shape (After skipping the header) : %s', data.shape)

    # shuffle the data
    np.random.shuffle(data)

    # get the labels
    labels = data[:, -1:]
    data = data[:, :-1]

    # convert string to float32
    data = data.astype('float32')

    # convert labels to int class
    classnames, labels = np.unique(labels, return_inverse=True)

    logger.debug('First 10 rows: \n%s', data[:10, ])
    logger.debug('First 10 lables: \n%s', labels[:10, ])

    logger.debug('Labels class names: \n%s', classnames)

    return data, labels, classnames


def train_with_neural_network(x_train, y_train_label, x_test, y_test_label):
    # (1) Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # (2) convert labels to hot encoded
    # we need to convert the label to hot encoded format
    y_train = to_categorical(y_train_label)
    y_test = to_categorical(y_test_label)


    # (3) build the neural network model
    model = Sequential([
        Dense(5, input_shape=(4,), activation='relu'),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax'),
    ])

    model.summary(print_fn=lambda line: logger.info(line))

    # compile the model
    learning_rate = 0.001

    logger.info('Compile the model with Learning rate = %f', learning_rate)
    model.compile(Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=5, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info("Neural Network: %.2f%%" % (test_acc * 100))


def train_with_linear_multinomial_naive_bayes(x_train, y_train, x_test, y_test):
    # Naive bayes classifier

    # (1) Normalize the data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    logger.info('Multinomial Naive Bayes\t%.2f%%', accuracy * 100)


def train_with_linear_multinomial_naive_bayes_with_normalization(x_train, y_train, x_test, y_test):
    # Naive bayes classifier

    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    logger.info('Multinomial Naive Bayes with Normalization\t%.2f%%', accuracy * 100)


def train_with_linear_svm(x_train, y_train, x_test, y_test):

    # (1) Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Training SVM
    model = LinearSVC()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    logger.info('Linear SVC \t%.2f%%', accuracy * 100)


def train_with_poly_svm(x_train, y_train, x_test, y_test):

    # (1) Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Training SVM
    model = SVC(gamma=0.1, kernel='poly', random_state=0)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    logger.info('SVM Classifier with gamma = 0.1; Kernel = Polynomial \t%.2f%%', accuracy * 100)


def train_with_decision_tree(x_train, y_train, x_test, y_test):

    # Training Decision tree
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    logger.info('Decision Tree \t%.2f%%', accuracy * 100)


if __name__ == '__main__':
    logger.info('build dictionary of all emails ... ')
    data, labels, classnames = read_iris_data()

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.35, random_state=0)

    train_with_neural_network(x_train, y_train, x_test, y_test)

    train_with_linear_multinomial_naive_bayes(x_train, y_train, x_test, y_test)

    train_with_linear_multinomial_naive_bayes_with_normalization(x_train, y_train, x_test, y_test)

    train_with_linear_svm(x_train, y_train, x_test, y_test)

    train_with_poly_svm(x_train, y_train, x_test, y_test)

    train_with_decision_tree(x_train, y_train, x_test, y_test)
