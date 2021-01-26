# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# https://www.kaggle.com/fk0728/feature-engineering-with-sklearn-pipelines
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#impute-the-missing-data-and-score
# https://scikit-learn.org/stable/modules/impute.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
#https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold

# https://scikit-lego.readthedocs.io/en/latest/index.html

from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import Ridge, LinearRegression, Lasso, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.tree import ExtraTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, f1_score, roc_auc_score, mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# import keras
# from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
# from keras.models import Sequential
# from keras.utils import to_categorical
# from keras.optimizers import SGD 
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import np_utils
# from keras.layers import LSTM
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# # from keras.layers import Dropout
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.client import device_lib

import os
import pandas as pd
import numpy as np
from io import BytesIO
from io import TextIOWrapper
import zipfile
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, normaltest, kstest
# from scipy.stats import normaltest
import warnings
from IPython.display import Image
# from patsy import PatsyModel, PatsyTransformer
import itertools
from numpy.polynomial.polynomial import polyfit
from random import randint
from math import sqrt

warnings.filterwarnings('ignore')
sns.set()
pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def get_dataset(dictionary):
    
    """
    https://scikit-learn.org/0.16/datasets/index.html
    https://scikit-learn.org/stable/datasets/index.html
    """

    for values in dictionary.values():
        #key = pd.DataFrame.from_dict(dictionary.values)
        if np.isscalar(values):
            pass
        else:
            #print(pd.DataFrame.from_dict(values))
            feature_names = dictionary["feature_names"]
            data = pd.DataFrame(dictionary["data"], columns=feature_names)
            target = pd.DataFrame(dictionary["target"], columns=["TARGET"])
            output = pd.concat([data,target],axis=1)
        
        return output


#for dataset in [load_boston(), load_iris(), load_diabetes()]:
#print(get_dataset(dataset)[:5])

def get_current_working_directory():

        """
        :return:
        """
        
        current_path = os.getcwd()

        return current_path


def change_current_working_directory(directory):
    """
    :param directory:
    :return:
    """
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("\n" + "Directory Does Not Exists. Working Directory Have Not Been Changed." + "\n")

    current_path = str(os.getcwd())
    
    return current_path

def get_list_of_files_from_directory(directory):
    """
    :param directory:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        list_of_files.append(item)

    return list_of_files

def get_list_of_zip_files(directory):
    """
    :param directory:
    :return:
    """
    os.chdir(directory)
    zip_files = []

    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename.endswith(".zip"):
                zip_files.append(filename)

    return zip_files

def get_list_of_files_by_extension(directory, extension):
    """
    :param directory:
    :param extension:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        if item.endswith("." + extension):
            list_of_files.append(item)

    return list_of_files

def unzip_files(directory, output_directory, zip_file_name):
    """
    :param input_directory:
    :param output_directory:
    :return:
    """

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    print("Unpacked " + str(zip_file_name) + " to: " + str(output_directory) + "\n")
    
def get_list_of_files_by_extension(directory, extension):
    """
    :param directory:
    :param extension:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        if item.endswith("." + extension):
            list_of_files.append(item)

    return list_of_files

def count_unique_values(dataframe, variables):
    """
    """
    for column in variables:
        count_unique = dataframe[str(column)].value_counts()
        count_null = pd.Series(dataframe[str(column)].isnull().sum(),index=["nan"])
        count_unique = count_unique.append(count_null, ignore_index=False)
        
        print(column + " count distinct:")
        print(count_unique)
        print()

def visualise_floats(dataframe, variables):
    """
    """
    for column in variables:
        ax = sns.distplot(dataframe[column].dropna(), fit=norm)
        ax.set_title("Histogram of " + str(column) + " before imputation")
        ax.set_xlabel(str(column))
        ax.set_ylabel("Frequency Rate")
        fig = plt.figure()
        
        res = stats.probplot(dataframe[column], plot=plt)
        fig = plt.figure()
        
#         target_column = pd.DataFrame(dataframe.iloc[:,-1])
#         test_output = pd.merge(target_column, dataframe[variables], left_index=True, right_index=True)
        
#         ax = sns.jointplot(x=column, y=target, data=dataframe, kind='reg', marker="+", color="b")
#         ax.fig.suptitle("Scatter plot of " + str(column) + "vs. " + target + " before imputation")
#         plt.figure()

def choose_imputer_and_visualise_floats(dataframe, variables, target, imputer=None, strategy=None, weights=None):
    """ 
    :SimpleImputer:
    :IterativeImputer:
    :KNNImputer:
    
    :SimpleImputer strategy:
    "mean"
    "median"
    "most_frequent"
    "constant"
    
    :KNNImputer weights:
    "uniform"
    "distance"
    "callable" 
    """
    
    #print("$ Counts before Imputation:")
    #for column in variables:
    #    print(count_unique_values(dataframe, column))
    #    print()
    
    if imputer == None:
        output = pd.DataFrame(dataframe.fillna(0), columns=variables)
        
    elif imputer == SimpleImputer and strategy != None:
        SI = SimpleImputer(missing_values=np.nan, strategy=str(strategy))
        SI.fit(dataframe[variables])
        output = pd.DataFrame(SI.transform(dataframe[variables]), columns=variables)
        
    elif imputer ==  IterativeImputer:
        II = IterativeImputer(max_iter=10, random_state=0)
        II.fit(dataframe[variables])
        output = pd.DataFrame(II.transform(dataframe[variables]), columns=variables)
        
    elif imputer == KNNImputer and weights != None:
        KNNI = KNNImputer(missing_values=np.nan, weights=str(weights), add_indicator=False)
        output = pd.DataFrame(KNNI.fit_transform(dataframe[variables]), columns=variables)
        
    else:
        output = "error"
    
    #print("$ Counts after Imputation:")
    #for column in output.columns:
    #    count_unique = output[column].value_counts()
    #    print(column)
    #    print(count_unique)
    #    print()
        
    
    for column in variables:
        ax = sns.distplot(output[column], fit=norm)
        ax.set_title("Histogram of " + str(column) + " after imputation")
        ax.set_xlabel(str(column))
        ax.set_ylabel("Frequency Rate")
        fig = plt.figure()
        
        res = stats.probplot(output[column], plot=plt)
        fig = plt.figure()
        
        if target != None:
        
            target_column = pd.DataFrame(dataframe.iloc[:,-1])
            test_output = pd.merge(target_column, output, left_index=True, right_index=True)
            ax = sns.jointplot(x=column, y=target, data=test_output, kind='reg', marker="+", color="b")
            ax.fig.suptitle("Scatter plot of " + str(column) + "vs. " + target + " after imputation")
            plt.figure()


    return output

def choose_imputer_and_visualise_categories(dataframe, variables, imputer=None, strategy=None, weights=None):
    """ 
    :SimpleImputer:
    :IterativeImputer:
    :KNNImputer:
    
    :SimpleImputer strategy:
    "mean"
    "median"
    "most_frequent"
    "constant"
    
    :KNNImputer weights:
    "uniform"
    "distance"
    "callable" 
    """
    
    #print("$ Counts before Imputation:")
    #for column in variables:
    #    print(count_unique_values(dataframe, column))
    #    print()
    
    if imputer == None:
        output = pd.DataFrame(dataframe.fillna(0), columns=variables)
        
    elif imputer == SimpleImputer and strategy != None:
        SI = SimpleImputer(missing_values=np.nan, strategy=str(strategy))
        SI.fit(dataframe[variables])
        output = pd.DataFrame(SI.transform(dataframe[variables]), columns=variables)
        
    elif imputer ==  IterativeImputer:
        II = IterativeImputer(max_iter=10, random_state=0)
        II.fit(dataframe[variables])
        output = pd.DataFrame(II.transform(dataframe[variables]), columns=variables)
        
    elif imputer == KNNImputer and weights != None:
        KNNI = KNNImputer(missing_values=np.nan, weights=str(weights), add_indicator=False)
        output = pd.DataFrame(KNNI.fit_transform(dataframe[variables]), columns=variables)
        
    else:
        output = "error"
        
    #print("$ Counts after Imputation:")
    #for column in range(len(output.columns)):
    #    count_unique = output[column].value_counts()
    #    print(count_unique)
    #    print()
        
    for column in variables:
        ax = sns.countplot(output[column], palette="Paired")
        ax.set_title("Bar plot of " + str(column) + " after imputation")
        ax.set_xlabel(str(column))
        fig = plt.figure()
            
    return output

def add_deviation_features(dataframe, variables_floats, variables_objects):
    
    """
    feature numeric
    category object
    """
    
    data = []

    #categories = pd.DataFrame(dataframe.select_dtypes(include=['object'])).columns
    categories = variables_objects
    #features = pd.DataFrame(dataframe.select_dtypes(include=['float64'])).columns
    features = variables_floats
    
    for category in categories:
        for feature in features:
            category_feature = str(category) + "_DEVIATION_" + str(feature)

            category_gb = dataframe.groupby(category)[feature]
            category_mean = category_gb.transform(lambda x: x.mean())
            category_std = category_gb.transform(lambda x: x.std())
            
            deviation_feature = ((dataframe[feature] - category_mean) / category_std).rename(category_feature)
            data.append(deviation_feature)
    
    output = pd.DataFrame(data).T
    dataframe = pd.concat([dataframe, output], axis=1)
    
    return dataframe

# Deep Learning

def get_n_last_days(df, series_name, n_days):
    """
    Extract last n_days of an hourly time series
    """
    
    return df[series_name][-(n_days):]

def plot_n_last_days(df, series_name, n_days):
    """
    Plot last n_days of an hourly time series 
    """
    plt.figure(figsize = (10,5))   
    plt.plot(get_n_last_days(df, series_name, n_days), 'k-')
    plt.title('{0} Air Quality Time Series - {1} days'.format(series_name, n_days))
    plt.xlabel('Recorded Hour')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    
def get_keras_format_series(series):
    """
    Convert a series to a numpy array of shape 
    [n_samples, time_steps, features]
    """
    
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1)

def get_train_test_data(df, series_name, series_days, input_days, test_days, sample_gap=3):
    """
    Utility processing function that splits an hourly time series into 
    train and test with keras-friendly format, according to user-specified
    choice of shape.    
    
    arguments
    ---------
    df (dataframe): dataframe with time series columns
    series_name (string): column name in df
    series_days (int): total days to extract
    input_days (int): length of sequence input to network 
    test_days (int): length of held-out terminal sequence
    sample_gap (int): step size between start of train sequences; default 5
    
    returns
    ---------
    tuple: train_X, test_X_init, train_y, test_y     
    """
    
    forecast_series = get_n_last_days(df, series_name, series_days).values # reducing our forecast series to last n days

    train = forecast_series[:-test_days] # training data is remaining days until amount of test_days
    test = forecast_series[-test_days:] # test data is the remaining test_days

    train_X, train_y = [], []

    # range 0 through # of train samples - input_days by sample_gap. 
    # This is to create many samples with corresponding
    for i in range(0, train.shape[0]-input_days, sample_gap): 
        train_X.append(train[i:i+input_days]) # each training sample is of length input hours
        train_y.append(train[i+input_days]) # each y is just the next step after training sample

    train_X = get_keras_format_series(train_X) # format our new training set to keras format
    train_y = np.array(train_y) # make sure y is an array to work properly with keras
    
    # The set that we had held out for testing (must be same length as original train input)
    test_X_init = test[:input_days] 
    test_y = test[input_days:] # test_y is remaining values from test set
    
    return train_X, test_X_init, train_y, test_y

def fit_SimpleRNN(train_X, train_y, cell_units, epochs):
    """
    Fit Simple RNN to data train_X, train_y 
    
    arguments
    ---------
    train_X (array): input sequence samples for training 
    train_y (list): next step in sequence targets
    cell_units (int): number of hidden units for RNN cells  
    epochs (int): number of training epochs   
    """

    # initialize model
    model = Sequential() 
    
    # construct an RNN layer with specified number of hidden units
    # per cell and desired sequence input format 
    model.add(SimpleRNN(cell_units, input_shape=(train_X.shape[1],1)))
    
    # add an output layer to make final predictions 
    model.add(Dense(1))
    
    # define the loss function / optimization strategy, and fit
    # the model with the desired number of passes over the data (epochs) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_y, epochs=epochs, shuffle=False, verbose=1)
    
    return model

def predict(X_init, n_steps, model):
    """
    Given an input series matching the model's expected format,
    generates model's predictions for next n_steps in the series      
    """
    
    X_init = X_init.copy().reshape(1,-1,1)
    preds = []
    
    # iteratively take current input sequence, generate next step pred,
    # and shift input sequence forward by a step (to end with latest pred).
    # collect preds as we go.
    for _ in range(n_steps):
        pred = model.predict(X_init)
        preds.append(pred)
        X_init[:,:-1,:] = X_init[:,1:,:] # replace first 11 values with 2nd through 12th
        X_init[:,-1,:] = pred # replace 12th value with prediction
    
    preds = np.array(preds).reshape(-1,1)
    
    return preds

def predict_and_plot(X_init, y, model, title):
    """
    Given an input series matching the model's expected format,
    generates model's predictions for next n_steps in the series,
    and plots these predictions against the ground truth for those steps 
    
    arguments
    ---------
    X_init (array): initial sequence, must match model's input shape
    y (array): true sequence values to predict, follow X_init
    model (keras.models.Sequential): trained neural network
    title (string): plot title   
    """
    
    y_preds = predict(X_init, n_steps=len(y), model=model) # predict through length of y
    # Below ranges are to set x-axes
    start_range = range(1, test_X_init.shape[0]+1) #starting at one through to length of test_X_init to plot X_init
    predict_range = range(test_X_init.shape[0], test_days)  #predict range is going to be from end of X_init to length of test_days
    
    #using our ranges we plot X_init
    plt.plot(start_range, test_X_init)
    #and test and actual preds
    plt.plot(predict_range, test_y, color='orange')
    plt.plot(predict_range, y_preds, color='teal', linestyle='--')
    
    plt.title(title)
    plt.legend(['Initial Series','Target Series','Predictions'])
    
def fit_LSTM(train_X, train_y, cell_units, epochs):
    """
    Fit LSTM to data train_X, train_y 
    
    arguments
    ---------
    train_X (array): input sequence samples for training 
    train_y (list): next step in sequence targets
    cell_units (int): number of hidden units for LSTM cells  
    epochs (int): number of training epochs   
    """
    
    # initialize model
    model = Sequential() 
    
    # construct a LSTM layer with specified number of hidden units
    # per cell and desired sequence input format 
    model.add(LSTM(cell_units, input_shape=(train_X.shape[1],1))) #,return_sequences= True))
    #model.add(LSTM(cell_units_l2, input_shape=(train_X.shape[1],1)))
    
    # add an output layer to make final predictions 
    model.add(Dense(1))
    
    # define the loss function / optimization strategy, and fit
    # the model with the desired number of passes over the data (epochs) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_y, epochs=epochs, shuffle=False, verbose=1)
    
    return model

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    """
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    ax.set_xlabel('Predicted Label') 
    ax.set_ylabel('True Label')
    
    return im, cbar

def annotate_heatmap(im, data=None, fmt="d", threshold=None):
    """
    A function to annotate a heatmap.
    """

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment="center",
                                 color="white" if data[i, j] > thresh else "black")
            texts.append(text)

    return texts