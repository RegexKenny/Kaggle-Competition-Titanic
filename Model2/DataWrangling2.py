# This file is for the Kaggle Competition Titanic
# This file is to perform the data wrangling based on A Data Science Framework: To Achieve 99% Accuracy
# Author: Kenny Hou
# Created at EY office desk 29.033C, 30/08/2019 9:54 AM

import pandas as pd


# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Visualization
import matplotlib as mpl
import matplotlib.pylab as pylab
import seaborn as sns
# from pandas.tools.plotting import scatter_matrix

# Configure Visualization Defaults
# mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8


# Check data quality including data correctness, data integrity and data type
def data_info(raw, val, copy):
    print(raw.info())
    print('-' * 10)
    print('Train columns with null values:\n', copy.isnull().sum())
    print('-' * 10)
    print('Test/Validation columns with null values:\n', val.isnull().sum())
    print('-' * 10)
    print(raw.describe(include='all'))


# 1. Clean data
# Complete or delete missing values in train and test/validation dataset
def clean_data(cleaner):
    drop_column = ['Cabin', 'Ticket']
    for dataset in cleaner:
        # complete missing age with median
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

        # complete embarked with mode. The mode of a set of values is the value that appears most often.
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

        # complete missing fare with median
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

        dataset.drop(drop_column, axis=1, inplace=True)
    cleaner[0].drop(['PassengerId'], axis=1, inplace=True)

    # print(copy.isnull().sum())
    # print('_' * 10)
    # print(cleaner[1].isnull().sum())

    return cleaner


# 2. Create new features
# Feature Engineering for train and test/validation dataset
def create_new_features(cleaner):
    # state_min = 10

    for dataset in cleaner:
        # Discrete Variables
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 1
        dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
        dataset['Title'] = dataset['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
                                                     'Sir', 'Jonkheer', 'Dona'], 'Misc')

    # title_names = (cleaner[0]['Title'].value_counts() < state_min)
    # print(title_names)

    # cleaner[0]['Title'] = cleaner[0]['Title'].apply(lambda x: 'Misc' if title_names.loc[x] else x)
    # cleaner[1]['Title'] = cleaner[1]['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
    #                                                    'Sir', 'Jonkheer', 'Dona'], 'Misc')
    # print(cleaner[1]['Title'].value_counts())
    # print('-' * 10)

    return cleaner


# 3. Convert Formats
# convert objects to category using Label Encoder for train and test/validation dataset
def convert_format(cleaner):
    label = LabelEncoder()
    for dataset in cleaner:
        dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
        dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
        dataset['Title_Code'] = label.fit_transform(dataset['Title'])
        dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
        dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    # print(cleaner[0].sample(10))
    # print('-' * 10)
    # print(cleaner[1].sample(10))
    return cleaner


def drop_useless_feature(cleaner):
    drop_column = ['FamilySize', 'SibSp', 'Parch', 'Age', 'Fare', 'Name', 'Sex', 'Embarked', 'Title',
                   'FareBin', 'AgeBin']
    for dataset in cleaner:
        dataset.drop(drop_column, axis=1, inplace=True)

    return cleaner


def write_csv(cleaner):
    cleaner[0].to_csv('train_wrangled.csv')
    cleaner[1].to_csv('test_wrangled.csv')


if __name__ == '__main__':

    data_raw = pd.read_csv('../train.csv')
    data_val = pd.read_csv('../test.csv')
    data_cleaner = [data_raw, data_val]

    # data_info(data_raw, data_val, data_copy)
    data_cleaner = clean_data(data_cleaner)
    data_cleaner = create_new_features(data_cleaner)
    data_cleaner = convert_format(data_cleaner)
    data_cleaner = drop_useless_feature(data_cleaner)
    write_csv(data_cleaner)


