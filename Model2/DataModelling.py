# This file is for the Kaggle Competition Titanic
# This file is to perform the data wrangling based on A Data Science Framework: To Achieve 99% Accuracy
# Author: Kenny Hou
# Created at EY office desk 29.033C, 30/08/2019 3:40 PM

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
# import xgboost import XGBClassifier  # need to figure out how to install it

data_train = pd.read_csv('train_wrangled.csv')
data_test = pd.read_csv('test_wrangled.csv')

X_train = data_train.drop('Survived', axis=1)
Y_train = data_train['Survived']

# machine learning algorithm initialization
MLA = [
    # Ensemble
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # Linear model
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Naive Bayes,
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbour
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
]

# Cross-validation Splitter
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

# MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Train Accuracy Mean',
               'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# Index through MLA
row_index = 0
for algorithm in MLA:
    MLA_name = algorithm.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(algorithm.get_params())

    cv_results = model_selection.cross_validate(algorithm, X_train, Y_train, cv=cv_split, return_train_score=True)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    algorithm.fit(X_train, Y_train)
    Predict = algorithm.predict(data_test)

    row_index += 1

MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare.to_csv('Algorithm comparison result.csv')
# print(MLA_compare)

# Plat the accuracy result
# sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='m')
#
# prettify
# plt.title('Machine Learning Algorithm Accuracy Score \n')
# plt.xlabel('Accuracy Score (%)')
# plt.ylabel('Algorithm')


