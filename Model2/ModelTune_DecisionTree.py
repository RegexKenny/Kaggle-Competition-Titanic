# This file is for the Kaggle Competition Titanic
# This file is to perform the data wrangling based on A Data Science Framework: To Achieve 99% Accuracy
# Author: Kenny Hou
# Created at EY office desk 29.037G, 02/09/2019 11:46 AM


import pandas as pd
from sklearn import model_selection, feature_selection
from sklearn import tree

data_train = pd.read_csv('train_wrangled.csv')
data_test = pd.read_csv('test_wrangled.csv')

X_train = data_train.drop('Survived', axis=1)
X_train = X_train.drop(data_train.columns[[0]], axis=1)
Y_train = data_train['Survived']
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [2, 4, 6, 8, 10, None],
              'random_state': [0]}

def model_tune(trainX, trainY):
    # Basic model
    dtree = tree.DecisionTreeClassifier(random_state=0)

    base_result = model_selection.cross_validate(dtree, trainX, trainY, cv=cv_split, return_train_score=True)
    dtree.fit(trainX, trainY)

    # Tune model
    tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc',
                                              cv=cv_split, return_train_score=True)
    tune_model.fit(trainX, trainY)

    return dtree, base_result, tune_model


def model_tune_print(dtree, base_result, tune_model):
    print('Before DT Parameters: ', dtree.get_params())
    print('Before DT Training w/bin score mean: {:.2f}'.format(base_result['train_score'].mean() * 100))
    print('Before DT Test w/bin score mean: {:.2f}'.format(base_result['test_score'].mean() * 100))
    print('Before DT Test 2/bin score 3*std: +/- {:.2f}'.format(base_result['test_score'].std() * 100 * 3))
    print('-' * 10)

    print('After DT Parameters:', tune_model.best_params_)
    print('After DT Training 2/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score']
                                                              [tune_model.best_index_] * 100))
    print('After DT Test 2/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score']
                                                          [tune_model.best_index_] * 100))
    print('After DT Test 2/bin score 3*std: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score']
                                                               [tune_model.best_index_] * 100 * 3))
    print('-' * 10)


def select_feature(base_result, tune_model, trainX, trainY):
    print('Before DT RFE Training Shape Old:', trainX.shape)
    print('Before DT RFE Training Columns Old:', trainX.columns.values)
    print('Before DT RFE Training w/bin score mean: {:.2f}'.format(base_result['train_score'].mean() * 100))
    print('Before DT RFE Test w/bin score mean: {:.2f}'.format(base_result['test_score'].mean() * 100))
    print('Before DT RFE Test w/bin score 3*std: +- {:.2f}'.format(base_result['test_score'].std() * 100 * 3))
    print('-' * 100)

    # feature selection
    dtree_rfe = feature_selection.RFECV(dtree, step=1, scoring='accuracy', cv=cv_split)
    dtree_rfe.fit(trainX, trainY)

    X_rfe = trainX.columns.values[dtree_rfe.get_support()]  # get true for all features
    rfe_result = model_selection.cross_validate(dtree, trainX[X_rfe], trainY, cv=cv_split, return_train_score=True)

    print('After DT RFE Training Shape New:', trainX[X_rfe].shape)
    print('After DT RFE Training Columns New:', X_rfe)
    print('After DT RFE Training w/bin score mean: {:.2f}'.format(rfe_result['train_score'].mean() * 100))
    print('After DT RFE Test w/bin score mean: {:.2f}'.format(rfe_result['test_score'].mean() * 100))
    print('After DT RFE Test w/bin score 3*std: +- {:.2f}'.format(rfe_result['test_score'].std() * 100 * 3))
    print('-' * 100)

    rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid,
                                                  scoring='roc_auc', cv=cv_split, return_train_score=True)
    rfe_tune_model.fit(trainX[X_rfe], trainY)
    print('After DT RFE Tuned Parameters:', rfe_tune_model.best_params_)
    print('After DT RFE Tuned Training w/bin score mean: {:.2f}'
          .format(rfe_tune_model.cv_results_['mean_train_score'][rfe_tune_model.best_index_] * 100))
    print('After DT RFE Tuned Test w/bin score mean: {:.2f}'
          .format(rfe_tune_model.cv_results_['mean_test_score'][rfe_tune_model.best_index_] * 100))
    print('After DT RFE Tuned Test w/bin score 3*std: +- {:.2f}'
          .format(rfe_tune_model.cv_results_['std_test_score'][rfe_tune_model.best_index_] * 100 * 3))
    print('-' * 100)


if __name__ == '__main__':
    dtree, base_result, tune_model = model_tune(X_train, Y_train)
    # model_tune_print(dtree, base_result, tune_model)
    select_feature(base_result, tune_model, X_train, Y_train)



