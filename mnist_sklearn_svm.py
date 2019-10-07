import logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from mnist_reader import load_mnist_dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train_svm_model(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    return accuracy_score(y_test, y_predict)


if __name__ == '__main__':
    models = {}

    # load MNIST dataset
    logger.debug('loading MNIST dataset')
    x_train, y_train, x_test, y_test = load_mnist_dataset()

    # type convert to float64
    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')

    # Standardize the data
    standard_scaler = StandardScaler()
    logger.debug('standardize the data')
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.transform(x_test)

    # HINT: Linear SVM fails to converge with the following error
    # ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
    logger.debug('SVM Linear Classifier')
    clf = LinearSVC()
    models.update({'SVM Linear Classifier': clf})

    logger.debug('SVM Classifier with gamma = 0.1; Kernel = Polynomial')
    clf = SVC(gamma=0.1, kernel='poly', random_state=0)
    models.update({'SVM Classifier with gamma = 0.1; Kernel = Polynomial': clf})

    logger.debug('SVM Classifier with gamma = 0.1; Kernel = rbf')
    clf = SVC(kernel='rbf', C=10.0, gamma=0.1)
    models.update({'SVM Classifier with gamma = 0.1; Kernel = rbf': clf})

    logger.debug('SVM Classifier with GridSearchCV')
    # Grid Search
    # Parameter Grid
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    # Make grid search classifier
    clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)
    models.update({"'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]": clf_grid})

    logger.info('SVM Type\t\t\t\tAccuracy')
    for (key, value) in models.items():
        accuracy = train_svm_model(value, x_train, y_train, x_test, y_test)
        logger.info('%s\t%.2f%%', key, accuracy * 100)
