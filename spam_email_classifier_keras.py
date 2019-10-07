import logging
import os
from collections import Counter

import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def build_dictionary_of_all_emails():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, 'data', 'emails')

    logger.debug('build dictionary of all emails.')
    sub_dirs = ['ham', 'spam']

    words = []
    for sub_dir in sub_dirs:
        email_dir = os.path.join(root_dir, sub_dir)

        # iterate on email_dir
        for file_name in os.listdir(email_dir):
            logger.debug('reading %s', file_name)
            file_path = os.path.join(email_dir, file_name)
            with open(file_path, 'r', encoding='latin-1') as file:
                words.extend(file.read().split(' '))

    logger.debug('Done reading all emails, Total words = %d', len(words))

    # remove non alpha numeric words from the list
    words_itr = filter(lambda word: word.isalpha(), words)
    # iterator to list
    words = list(words_itr)
    logger.debug('After isalpha filtering, Total words = %d', len(words))

    # Convert words list to dictionary
    word_dict = Counter(words)
    logger.debug('Unique words = %d', len(word_dict))

    return word_dict.most_common(20000)


def build_dataset(word_dict):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, 'data', 'emails')

    logger.debug('building feature set of the data.')
    sub_dirs = ['ham', 'spam']

    feature_set = []
    labels = []

    for sub_dir in sub_dirs:
        email_dir = os.path.join(root_dir, sub_dir)

        # iterate on email_dir
        for file_name in os.listdir(email_dir):
            logger.debug('scanning %s', file_name)
            file_path = os.path.join(email_dir, file_name)

            data = []
            with open(file_path, 'r', encoding='latin-1') as file:
                words = file.read().split(' ')
                for key, value in word_dict:
                    data.append(words.count(key))
            feature_set.append(data)
            labels.append(1 if 'spam' in file_name else 0)

    # convert the feature_set and labels to numpy array
    dataset = np.array(feature_set)
    labels = np.array(labels)

    logger.debug('data-set shape: %s', dataset.shape)
    logger.debug('labels shape: %s', labels.shape)

    # append labels as a last column of dataset
    dataset = np.hstack((dataset, labels[:, np.newaxis]))
    logger.debug('data-set shape after appending the labels: %s', dataset.shape)
    logger.debug('data-set after appending the labels, example ( first 5 rows ):\n %s', dataset[:5, ])

    # shuffle the data
    np.random.shuffle(dataset)
    logger.debug('data-set after shuffling, example ( first 5 rows ):\n %s', dataset[:5, ])

    x_train = dataset[:, 0:-1]
    y_train = np.ravel(dataset[:, -1:])

    logger.debug('x_train shape: %s', x_train.shape)
    logger.debug('y_train shape: %s', y_train.shape)
    logger.debug('x_train example ( first 5 rows ):\n %s', x_train[:5, ])
    logger.debug('y_train example ( first 5 rows ):\n %s', y_train[:5, ])

    return x_train, y_train


def split_dataset(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.35, random_state=0)

    logger.debug('data shape: %s', data.shape)
    logger.debug('labels shape: %s', labels.shape)

    logger.debug('x_train shape: %s', x_train.shape)
    logger.debug('y_train shape: %s', y_train.shape)

    logger.debug('x_test shape: %s', x_test.shape)
    logger.debug('y_test shape: %s', y_test.shape)

    return x_train, y_train, x_test, y_test


def train_with_neural_network(x_train, y_train, x_test, y_test):
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
        Dense(20000, input_shape=(20000,), activation='relu'),
        Dense(1000, activation='relu'),
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

if __name__ == '__main__':
    logger.info('build dictionary of all emails ... ')
    word_dict = build_dictionary_of_all_emails()

    logger.info('build data-set of this dictionary ... ')
    x, y = build_dataset(word_dict)

    logger.info('splitting the data-set into train and test ... ')
    x_train, y_train, x_test, y_test = split_dataset(x, y)

    # logger.info('train with neural network ... ')
    # # very bad results
    # train_with_neural_network(x_train, y_train, x_test, y_test)

    logger.info('train with Multinomial Naive Bayes ... ')
    train_with_linear_multinomial_naive_bayes(x_train, y_train, x_test, y_test)

    logger.info('train with Multinomial Naive Bayes with Normalization ... ')
    train_with_linear_multinomial_naive_bayes_with_normalization(x_train, y_train, x_test, y_test)

    logger.info('train with Linear SVM ... ')
    train_with_linear_svm(x_train, y_train, x_test, y_test)

    logger.info('train with Polynomial SVM ... ')
    train_with_poly_svm(x_train, y_train, x_test, y_test)

    logger.info('train with Decision tree ... ')
    train_with_decision_tree(x_train, y_train, x_test, y_test)
