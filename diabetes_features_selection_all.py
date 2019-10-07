import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import train_all as tall

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def draw_stat_plots(df):
    # (1) Draw hist for each feature related to the label
    grid1 = sns.FacetGrid(df, col='Outcome')
    grid1.map(plt.hist, 'Age')

    grid2 = sns.FacetGrid(df, col='Outcome')
    grid2.map(plt.hist, 'Glucose')

    # and so on, we can plot all the relation between each feature and the label
    plt.show()

    # (2) for each feature, draw the relation with the label
    # set the background colour of the plot to white
    sns.set(style="whitegrid", color_codes=True)
    # setting the plot size for all plots
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    # create a countplot
    sns.countplot('Outcome', data=df, hue='Age')
    # Remove the top and down margin
    sns.despine(offset=10, trim=True)
    plt.show()

    # set the background colour of the plot to white
    sns.set(style="whitegrid", color_codes=True)
    # setting the plot size for all plots
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    # create a countplot
    sns.countplot('Outcome', data=df, hue='Glucose')
    # Remove the top and down margin
    sns.despine(offset=10, trim=True)
    plt.show()

    # (3) for each feature, draw the relation with the label
    # setting the plot size for all plots
    sns.set(rc={'figure.figsize': (16.7, 13.27)})
    # plotting the violinplot
    sns.violinplot(x="Outcome", y="Glucose", hue="Outcome", data=df)
    plt.show()

    # setting the plot size for all plots
    sns.set(rc={'figure.figsize': (16.7, 13.27)})
    # plotting the violinplot
    sns.violinplot(x="Outcome", y="Age", hue="Outcome", data=df)
    plt.show()


def analyze_the_data(df):
    # Analyze the data
    logger.debug("Pregnancies : %s", df['Pregnancies'].unique())
    logger.debug("Glucose : %s", df['Glucose'].unique())
    logger.debug("BloodPressure : %s", df['BloodPressure'].unique())
    logger.debug("SkinThickness : %s", df['SkinThickness'].unique())
    logger.debug("Insulin : %s", df['Insulin'].unique())
    logger.debug("BMI : %s", df['BMI'].unique())
    logger.debug("DiabetesPedigreeFunction : %s", df['DiabetesPedigreeFunction'].unique())
    logger.debug("Age : %s", df['Age'].unique())
    logger.debug("Outcome : %s", df['Outcome'].unique())


def select_best_features(features, X, Y):
    # choose some of the most important features with Recursive Feature Elimination
    model = MultinomialNB()
    rfe = RFE(model, 5)  # we have selected here 5 features
    fit = rfe.fit(X, Y)

    logger.debug('Features : %s', features)
    logger.debug('supported Features : %s', fit.support_)
    logger.debug('Features ranking : %s', fit.ranking_)

    features = np.array(features)
    mask = np.array(fit.support_)
    best_features = features[mask]
    logger.debug('Best features : %s', best_features)

    selected_rfe_features = pd.DataFrame({'Feature': features, 'Ranking': rfe.ranking_})
    selected_rfe_features = selected_rfe_features.sort_values(by='Ranking')

    logger.debug('Best features : \n%s', selected_rfe_features)



def select_best_features2(features, X, Y):
    # choose some of the most important features with Recursive Feature Elimination
    model = ExtraTreesClassifier()
    model.fit(X, Y)

    logger.debug('Features : %s', model.feature_importances_)

    logger.debug('Feature Name\t\tRank')
    for f, fi in zip(features, model.feature_importances_):
        logger.debug('%s\t\t%f', f, fi)

    selected_rfe_features = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    selected_rfe_features = selected_rfe_features.sort_values(by='Importance')

    logger.debug('Best features : \n%s', selected_rfe_features)

def select_best_features3(features, X, Y):
    clf_lr = LogisticRegression()
    rfecv = RFECV(estimator=clf_lr, step=1, cv=5, scoring='accuracy')
    rfecv = rfecv.fit(X, Y)

    logger.debug('Optimal number of features : %d', rfecv.n_features_)
    logger.debug('Best features : %s', features[rfecv.support_])


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(root_dir, 'data', 'pima-indians-diabetes-database', 'diabetes.csv')

    logger.debug('reading pima indians diabetes database data from: %s', data_file)
    df = pd.read_csv(data_file)

    # get the information and the meta data of the data frame
    # and check if there is a missed value
    logger.debug('data frame info\n : %s', df.info())
    logger.debug('sample data in data frame\n : %s', df.head())
    logger.debug('data frame shape : %s', df.shape)

    logger.debug('label value count : %s', df.Outcome.value_counts())

    # draw_stat_plots(df)

    # analyze_the_data(df)

    features = np.array(list(df)[:-1])
    data = df.values
    np.random.shuffle(data)
    X = data[:, 0:-1]
    Y = np.ravel(data[:, -1:])

    select_best_features(features, X, Y)
    select_best_features2(features, X, Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=0)


    logger.info('train with Multinomial Naive Bayes ... ')
    tall.train_with_linear_multinomial_naive_bayes(x_train, y_train, x_test, y_test)

    logger.info('train with Multinomial Naive Bayes with Normalization ... ')
    tall.train_with_linear_multinomial_naive_bayes_with_normalization(x_train, y_train, x_test, y_test)

    logger.info('train with Linear SVM ... ')
    tall.train_with_linear_svm(x_train, y_train, x_test, y_test)

    logger.info('train with Polynomial SVM ... ')
    tall.train_with_poly_svm(x_train, y_train, x_test, y_test)

    logger.info('train with Decision tree ... ')
    tall.train_with_decision_tree(x_train, y_train, x_test, y_test)

    logger.info('train with Logistic Regression  ... ')
    tall.train_with_logistic_regression(x_train, y_train, x_test, y_test)

    # logger.info('train with neural network ... ')
    # tall.train_with_neural_network(x_train, y_train, x_test, y_test, len(x_train[0]))
