import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(root_dir, 'data', 'california-housing-price-prediction', 'housing.csv')

    logger.debug('reading california housing price prediction data from: %s', data_file)
    df = pd.read_csv(data_file)

    # get the information and the meta data of the data frame
    # and check if there is a missed value
    logger.debug('data frame info\n : %s', df.info())
    logger.debug('data frame data types\n : %s', df.dtypes())
    logger.debug('sample data in data frame\n : %s', df.head())
    logger.debug('data frame shape : %s', df.shape)
    logger.debug('label value count : %s', df['median_house_value'].count())

    # Descriptive statistics for each column
    logger.debug('data statistics : \n%s', df.describe())

    df.hist(bins=50, figsize=(15, 15))
    plt.show()

    logger.debug('Data Analysis ...... ')
    features = np.array(list(df))
    for feature in features:
        logger.debug('%s, Values: \n%s', feature, df[feature].value_counts())
        logger.debug('----------------------------------------------------------------')

    # Treat incomplete data
    # replace missing data with median value (the “middle” number of total_bedrooms
    # in california check statistics definition for more details) total bedroom.
    # the median will not unbalanced the data-set
    median = df['total_bedrooms'].median()
    df['total_bedrooms'].fillna(median, inplace=True)

    # shuffle the data
    logger.debug('Shuffling the data ...')
    # df = df.sample(frac=1)
    df = shuffle(df, random_state=20)

    # split the data train and test
    df_length = len(df.index)
    logger.debug('# of row = %d', df_length)
    rows_count_for_test = int(df_length * 0.35)

    df_test, df_train = df[:rows_count_for_test], df[rows_count_for_test:]
    logger.debug('# of row for train = %d', len(df_train.index))
    logger.debug('# of row for test = %d', len(df_test.index))

    # split the train and test to data and labels
    index = np.argwhere(features == 'median_house_value')
    features = np.delete(features, index)
    logger.debug('Data Headers : \n%s', features)

    df_train_data, df_train_labels = df_train.drop('median_house_value', 1), df_train.drop(features, 1)
    df_test_data, df_test_labels = df_test.drop('median_house_value', 1), df_test.drop(features, 1)

    logger.debug('df_train_data = \n%s', df_train_data.head(n=3))
    logger.debug('df_test_data = \n%s', df_test_data.head(n=3))

    # Normalize, Standardize and Hash the features
    # each column has its own scaler
    # ============================================

    # longitude Column
    longitude_scaler = MinMaxScaler()
    train_col = df_train_data[['longitude']].values.astype(float)
    test_col = df_test_data[['longitude']].values.astype(float)
    df_train_data['longitude'] = longitude_scaler.fit_transform(train_col)
    df_test_data['longitude'] = longitude_scaler.transform(test_col)

    # latitude Column
    latitude_scaler = StandardScaler()
    train_col = df_train_data[['latitude']].values.astype(float)
    test_col = df_test_data[['latitude']].values.astype(float)
    df_train_data['latitude'] = latitude_scaler.fit_transform(train_col)
    df_test_data['latitude'] = latitude_scaler.transform(test_col)

    # housing_median_age Column
    housing_median_age_scaler = StandardScaler()
    train_col = df_train_data[['housing_median_age']].values.astype(float)
    test_col = df_test_data[['housing_median_age']].values.astype(float)
    df_train_data['housing_median_age'] = housing_median_age_scaler.fit_transform(train_col)
    df_test_data['housing_median_age'] = housing_median_age_scaler.transform(test_col)

    # total_rooms Column
    total_rooms_scaler = StandardScaler()
    train_col = df_train_data[['total_rooms']].values.astype(float)
    test_col = df_test_data[['total_rooms']].values.astype(float)
    df_train_data['total_rooms'] = total_rooms_scaler.fit_transform(train_col)
    df_test_data['total_rooms'] = total_rooms_scaler.transform(test_col)

    # total_bedrooms Column
    total_bedrooms_scaler = StandardScaler()
    train_col = df_train_data[['total_bedrooms']].values.astype(float)
    test_col = df_test_data[['total_bedrooms']].values.astype(float)
    df_train_data['total_bedrooms'] = total_bedrooms_scaler.fit_transform(train_col)
    df_test_data['total_bedrooms'] = total_bedrooms_scaler.transform(test_col)

    # population Column
    population_scaler = StandardScaler()
    train_col = df_train_data[['population']].values.astype(float)
    test_col = df_test_data[['population']].values.astype(float)
    df_train_data['population'] = population_scaler.fit_transform(train_col)
    df_test_data['population'] = population_scaler.transform(test_col)

    # households Column
    households_scaler = StandardScaler()
    train_col = df_train_data[['households']].values.astype(float)
    test_col = df_test_data[['households']].values.astype(float)
    df_train_data['households'] = households_scaler.fit_transform(train_col)
    df_test_data['households'] = households_scaler.transform(test_col)

    # median_income Column
    median_income_scaler = StandardScaler()
    train_col = df_train_data[['median_income']].values.astype(float)
    test_col = df_test_data[['median_income']].values.astype(float)
    df_train_data['median_income'] = median_income_scaler.fit_transform(train_col)
    df_test_data['median_income'] = median_income_scaler.transform(test_col)

    # ocean_proximity Column, it is a categorical data column
    # convert the categorical columns into numeric
    ocean_proximity_encoder = LabelEncoder()
    train_col = df_train_data['ocean_proximity'].values
    test_col = df_test_data['ocean_proximity'].values
    all_labels = np.concatenate([train_col, test_col])
    ocean_proximity_encoder.fit(all_labels)
    df_train_data['ocean_proximity'] = ocean_proximity_encoder.transform(train_col)
    df_test_data['ocean_proximity'] = ocean_proximity_encoder.transform(test_col)

    # the labels may be normalized as well
    labels_scaler = StandardScaler()
    # train_col = df_train_labels[['median_house_value']].values.astype(float)
    # test_col = df_test_labels[['median_house_value']].values.astype(float)
    # df_train_labels['median_house_value'] = labels_scaler.fit_transform(train_col)
    # df_test_labels['median_house_value'] = labels_scaler.transform(test_col)

    logger.debug('df_train_data after normalization = \n%s', df_train_data.head(n=3))
    logger.debug('df_test_data after normalization = \n%s', df_test_data.head(n=3))

    # finally save each of those scalars and encoders of each column to be used in predictions
    root_dir = os.path.dirname(os.path.abspath(__file__))
    scalers_dir_path = os.path.join(root_dir, 'saved_models', 'california-housing-price-prediction')

    # make sure that the dir exists
    if not os.path.exists(scalers_dir_path):
        os.makedirs(scalers_dir_path)

    scalers = [longitude_scaler, latitude_scaler, housing_median_age_scaler, total_rooms_scaler, total_bedrooms_scaler,
               population_scaler, households_scaler, median_income_scaler, ocean_proximity_encoder, labels_scaler]
    scaler_filenames = ['longitude_scaler.save', 'latitude_scaler.save', 'housing_median_age_scaler.save',
                        'total_rooms_scaler.save', 'total_bedrooms_scaler.save', 'population_scaler.save',
                        'households_scaler.save', 'median_income_scaler.save', 'ocean_proximity_encoder.save',
                        'labels_scaler.save']

    scalers_json = {'scaler': scalers, 'file_name': scaler_filenames}
    scalers_df = pd.DataFrame.from_dict(scalers_json)

    logger.debug('saving scalers into = %s', scalers_dir_path)
    logger.debug('scalers to be saved = %s', scalers_df.head(n=10))

    for index, row in scalers_df.iterrows():
        scalers_file = os.path.join(scalers_dir_path, row['file_name'])
        logger.debug('saving scaler into file: %s', scalers_file)
        joblib.dump(row['scaler'], scalers_file)

    # what is the most important features using Linear Regression
    # ============================================

    # Method 1: using RFE
    model = LinearRegression()
    rfe = RFE(model, 5)  # we have selected here 5 features

    logger.debug('RPF using Linear Regression ...')
    fit = rfe.fit(df_train_data.values, df_train_labels.values)

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

    # Method 2: using feature_importances_ in ExtraTreesClassifier
    # model = ExtraTreesClassifier()
    # model.fit(df_train_data.values, df_train_labels.values)
    #
    # logger.debug('Features : %s', model.feature_importances_)
    #
    # selected_rfe_features = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    # selected_rfe_features = selected_rfe_features.sort_values(by='Importance')
    #
    # logger.debug('Best features : \n%s', selected_rfe_features)

    # Method 3: using RFECV
    # model = LinearRegression()
    # rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
    # rfecv = rfecv.fit(df_train_data.values, df_train_labels.values)
    #
    # logger.debug('Optimal number of features : %d', rfecv.n_features_)
    # logger.debug('Best features : %s', features[rfecv.support_])

    x_train, y_train, x_test, y_test = df_train_data.values, df_train_labels.values, df_test_data.values, df_test_labels.values

    # Train the data
    # ======================

    # 1. instantiate the linear regression
    logger.info('Linear Regression Model ...')
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = LinearRegression(n_jobs=-1)
    scoring = 'neg_mean_squared_error'
    results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    logger.info("MSE: %.3f (%.3f)", results.mean(), results.std())
    # Train the model on training data
    model.fit(x_test, y_test)
    # Predict on test
    y_pred = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    logger.info('Mean Absolute Error: %.3f degrees.', round(np.mean(errors), 2))
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    logger.info('Accuracy: %.3f %%.', round(accuracy, 2))
    logger.info('-----------------------------------------------------------------')

    # 2. Decision Tree Regressor Model
    logger.info('Decision Tree Regressor Model ...')
    regressor = DecisionTreeRegressor(max_depth=9)
    scoring = 'neg_mean_squared_error'
    results = model_selection.cross_val_score(regressor, x_train, y_train, cv=10, scoring=scoring)
    logger.info("MSE: %.3f (%.3f)", results.mean(), results.std())
    # Train the model on training data
    regressor.fit(x_test, y_test)
    # Predict on test
    y_pred = regressor.predict(x_test)
    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    logger.info('Mean Absolute Error: %.3f degrees.', round(np.mean(errors), 2))
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    logger.info('Accuracy: %.3f %%.', round(accuracy, 2))
    logger.info('-----------------------------------------------------------------')

    # 2. Random Forest Regressor Model
    logger.info('Random Forest Regressor Model ...')
    fr_regressor = RandomForestRegressor(max_depth=9)
    scoring = 'neg_mean_squared_error'
    results = model_selection.cross_val_score(fr_regressor, x_train, y_train, cv=10, scoring=scoring)
    logger.info("MSE: %.3f (%.3f)", results.mean(), results.std())
    # Train the model on training data
    fr_regressor.fit(x_test, y_test)
    # Predict on test
    y_pred = fr_regressor.predict(x_test)
    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    logger.info('Mean Absolute Error: %.3f degrees.', round(np.mean(errors), 2))
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    logger.info('Accuracy: %.3f %%.', round(accuracy, 2))
    logger.info('-----------------------------------------------------------------')

    # implement neural network
    model = Sequential([
        Dense(9, input_shape=(9,), activation='relu'),
        Dense(20, activation='relu'),
        Dense(1, activation='linear'),
    ])

    model.summary(print_fn=lambda line: logger.info(line))

    # compile the model
    learning_rate = 0.001

    logger.info('Compile the model with Learning rate = %f', learning_rate)
    model.compile(Adam(lr=learning_rate), loss='mean_squared_error', metrics=['mse', 'mae'])

    # train the model
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=6000, verbose=2)

    logger.info(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # ynew = labels_scaler.inverse_transform(ynew)

    # very important notice, the labels also should be normalized
    # Xnew = np.array([[40, 0, 26, 9000, 8000]])
    # Xnew = scaler_x.transform(Xnew)
    # ynew = model.predict(Xnew)

    # # invert normalize
    # ynew = labels_scaler.inverse_transform(ynew)
    # Xnew = scaler_x.inverse_transform(Xnew)
    # print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
#
