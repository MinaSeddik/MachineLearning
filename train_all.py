import logging
import os

import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train_with_neural_network(x_train, y_train, x_test, y_test, num_of_features):
    # (1) Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # after normalize the data, we should save the scaler to use it in the test
    scaler_filename = 'email_spam_scaler.save'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(root_dir, 'saved_models', scaler_filename)

    joblib.dump(scaler, model_file_path)

    # And now to load...
    # scaler = joblib.load(scaler_filename)

    # (2) build the neural network model
    model = Sequential([
        Dense(num_of_features, input_shape=(num_of_features,), activation='relu'),
        Dense(num_of_features / 2, activation='relu'),
        Dense(1, activation='softmax'),
    ])

    model.summary(print_fn=lambda line: logger.info(line))

    # compile the model
    learning_rate = 0.001

    logger.info('Compile the model with Learning rate = %f', learning_rate)
    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=5, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info("Test accuracy: %.2f%%" % (test_acc * 100))

    # Save the model to disk.
    model_file_path = os.path.join(root_dir, 'saved_models', 'spam_email_keras_nn.h5')
    model.save(model_file_path)

    # Predict on the test labels.
    predictions = model.predict(x_test)

    # get the max class for each
    y_predict = np.argmax(predictions, axis=1)

    label_name = ['reguler', 'spam']
    logger.debug("Display the First 100 test data's predictions:")
    logger.debug('\tLabel\t\tPredicted Label:')
    for i in range(0, 100):
        logger.debug('\t%d\t\t%d', label_name[y_test[i]], label_name[y_predict[i]])


def train_with_logistic_regression(x_train, y_train, x_test, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    logger.info('Logistic Regression\t%.2f%%', accuracy * 100)


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

    # after normalize the data, we should save the scaler to use it in the test
    scaler_filename = 'email_spam_scaler.save'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(root_dir, 'saved_models', scaler_filename)

    joblib.dump(scaler, model_file_path)

    # And now to load...
    # scaler = joblib.load(scaler_filename)

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

    # after normalize the data, we should save the scaler to use it in the test
    scaler_filename = 'email_spam_scaler.save'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(root_dir, 'saved_models', scaler_filename)

    joblib.dump(scaler, model_file_path)

    # And now to load...
    # scaler = joblib.load(scaler_filename)

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
