# This file is for the Kaggle Competition Titanic
# This file is to perform the data wrangling based on A Data Science Framework: To Achieve 99% Accuracy
# Author: Kenny Hou
# Created at EY office desk 29.037G, 02/09/2019 2;55 PM


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import pandas as pd
from sklearn import model_selection
import time
import warnings

warnings.filterwarnings('ignore')

data_train = pd.read_csv('train_wrangled.csv')
data_test = pd.read_csv('test_wrangled.csv')

X_train = data_train.drop('Survived', axis=1)
X_train = X_train.drop(data_train.columns[[0]], axis=1)
Y_train = data_train['Survived']
p_id = data_test['PassengerId']
data_test = data_test.drop(data_test.columns[[0]], axis=1)
data_test = data_test.drop('PassengerId', axis=1)
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

vote_set = [
    # Ensemble methods
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    # Gaussian processes
    ('gpc', gaussian_process.GaussianProcessClassifier()),

    # Linear model
    ('lr', linear_model.LogisticRegression()),

    # Naive Bayes
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),

    # Knn
    ('knn', neighbors.KNeighborsClassifier()),

    # SVM
    ('svc', svm.SVC(probability=True)),
]

grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [0.1, 0.25, 0.5, 0.75, 1.0]
grid_learn = [0.01, 0.03, 0.05, 0.1, 0.25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, 0.03, 0.05, 0.10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]


def vote_comparison(vote_est, trainX, trainY):
    # Hard vote or majority rules
    hard_vote = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    hard_vote_cv = model_selection.cross_validate(hard_vote, trainX, trainY, cv=cv_split, return_train_score=True)
    hard_vote.fit(trainX, trainY)

    print('Hard Voting Training w/bin score mean: {:.2f}'.format(hard_vote_cv['train_score'].mean() * 100))
    print('Hard Voting Test w/bin score mean: {:.2f}'.format(hard_vote_cv['test_score'].mean() * 100))
    print('Hard Voting Test w/bin score 3*std: {:.2f}'.format(hard_vote_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # Soft vote or majority rules
    soft_vote = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    soft_vote_cv = model_selection.cross_validate(soft_vote, trainX, trainY, cv=cv_split, return_train_score=True)
    soft_vote.fit(trainX, trainY)

    print('Soft Voting Training w/bin score mean: {:.2f}'.format(soft_vote_cv['train_score'].mean() * 100))
    print('Soft Voting Test w/bin score mean: {:.2f}'.format(soft_vote_cv['test_score'].mean() * 100))
    print('Soft Voting Test w/bin score 3*std: {:.2f}'.format(soft_vote_cv['test_score'].std() * 100 * 3))
    print('-' * 10)


def run_alg(trainX, trainY, vote_est):
    grid_param = [
        [{
            # AdaBoostClassifier
            'n_estimators': grid_n_estimator,
            'learning_rate': grid_learn,
            'random_state': grid_seed
        }],
        [{
            # BaggingClassifier
            'n_estimators': grid_n_estimator,
            'max_samples': grid_ratio,
            'random_state': grid_seed
        }],
        [{
            # ExtraTreesClassifier
            'n_estimators': grid_n_estimator,
            'criterion': grid_criterion,
            'max_depth': grid_max_depth,
            'random_state': grid_seed
        }],
        [{
            # GradientBoostingClassifier
            'learning_rate': [0.05],
            'n_estimators': [300],
            'max_depth': grid_max_depth,
            'random_state': grid_seed
        }],
        [{
            # RandomForestClassifier
            'n_estimators': grid_n_estimator,
            'criterion': grid_criterion,
            'max_depth': grid_max_depth,
            'oob_score': [True],
            'random_state': grid_seed
        }],
        [{
            # Gaussian Process
            'max_iter_predict': grid_n_estimator,
            'random_state': grid_seed
        }],
        [{
            # Logistic Regression
            'fit_intercept': grid_bool,
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'random_state': grid_seed
        }],
        [{
            # Naive Bayes BernoulliNB
            'alpha': grid_ratio
        }],
        [{
            # Naive Bayes GaussianNB
        }],
        [{
            # Knn
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }],
        [{
            # SVC
            'C': [1, 2, 3, 4, 5],
            'gamma': grid_ratio,
            'decision_function_shape': ['ovo', 'ovr'],
            'probability': [True],
            'random_state': grid_seed
        }],
    ]

    start_total = time.perf_counter()
    for clf, param in zip(vote_est, grid_param):
        start = time.perf_counter()
        best_search = model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split,
                                                   return_train_score=True)
        best_search.fit(trainX, trainY)
        run = time.perf_counter() - start
        best_param = best_search.best_params_
        print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.
              format(clf[1].__class__.__name__, best_param, run))
        clf[1].set_params(**best_param)

    run_total = time.perf_counter() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total/60))
    print('-' * 10)


def hard_vote_tune(trainX, trainY, vote_est, test):
    grid_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, trainX, trainY, cv=cv_split, return_train_score=True)
    grid_hard.fit(trainX, trainY)
    # print('Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}'
    #       .format(grid_hard_cv['train_score'].mean() * 100))
    # print('Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}'
    #       .format(grid_hard_cv['test_score'].mean() * 100))
    # print('Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: {:.2f}'
    #       .format(grid_hard_cv['test_score'].std() * 100 * 3))
    # print('-' * 10)
    pre_result = grid_hard.predict(test)
    return pre_result  # 68.899%


def soft_vote_tune(trainX, trainY, vote_est, test):
    grid_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, trainX, trainY, cv=cv_split, return_train_score=True)
    grid_soft.fit(trainX, trainY)

    # print('Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}'
    #       .format(grid_soft_cv['train_score'].mean() * 100))
    # print('Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}'
    #       .format(grid_soft_cv['test_score'].mean() * 100))
    # print('Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: {:.2f}'
    #       .format(grid_soft_cv['test_score'].std() * 100 * 3))
    # print('-' * 10)
    pre_result = grid_soft.predict(test)
    return pre_result  # 73.205%


def dt_submit(trainX, trainY, test):
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [2, 4, 6, 8, 10, None],
                  'random_state': [0]}
    dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
                                      param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    dt.fit(trainX, trainY)
    print('Best Parameters: ', dt.best_params_)
    pre_result = dt.predict(test)
    return pre_result  # 73%


def bagging_submit(trainX, trainY, test):
    param_grid = {'n_estimators': grid_n_estimator,
                  'max_samples': grid_ratio,
                  'oob_score': grid_bool,
                  'random_state': grid_seed
                  }
    bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(),
                                      param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    bc.fit(trainX, trainY)
    print('Best parameters: ', bc.best_params_)
    pre_result = bc.predict(test)
    return pre_result  # 67.942%


def et_submit(trainX, trainY, test):
    param_grid = {'n_estimators': grid_n_estimator,
                  'criterion': grid_criterion,
                  'max_depth': grid_max_depth,
                  'random_state': grid_seed
                  }
    et = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(),
                                      param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    et.fit(trainX, trainY)
    print('Best parameters: ', et.best_params_)
    pre_result = et.predict(test)
    return pre_result  # 76.076%


def rf_submit(trainX, trainY, test):
    param_grid = {'n_estimators': grid_n_estimator,
                  'criterion': grid_criterion,
                  'max_depth': grid_max_depth,
                  'random_state': grid_seed
                  }
    rd = model_selection.GridSearchCV(ensemble.RandomForestClassifier(),
                                      param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    rd.fit(trainX, trainY)
    print('Best parameters: ', rd.best_params_)
    pre_result = rd.predict(test)
    return pre_result  # 71.291%


def ada_boosting_submit(trainX, trainY, test):
    param_grid = {'n_estimators': grid_n_estimator,
                  'learning_rate': grid_ratio,
                  'algorithm': ['SAMME', 'SAMME.R'],
                  'random_state': grid_seed
                  }
    ad = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(),
                                      param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    ad.fit(trainX, trainY)
    print('Best parameters: ', ad.best_params_)
    pre_result = ad.predict(test)
    return pre_result  # 69.856%


def gb_submit(trainX, trainY, test):
    param_grid = {'n_estimators': grid_n_estimator,
                  'learning_rate': grid_ratio,
                  'max_depth': grid_max_depth,
                  'random_state': grid_seed
                  }
    gb = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(),
                                      param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    gb.fit(trainX, trainY)
    print('Best parameters: ', gb.best_params_)
    pre_result = gb.predict(test)
    return pre_result  # 74.641%


# SGBClassifier, extreme boosting
# def eb_submit(trainX, trainY, test):
#     return pre_result  # %


def csv(p_id, predict):
    submission = pd.DataFrame({'PassengerId': p_id, 'Survived': predict})
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    # vote_comparison(vote_set, X_train, Y_train)
    # run_alg(X_train, Y_train, vote_set)
    # result = hard_vote_tune(X_train, Y_train, vote_set, data_test)
    result = soft_vote_tune(X_train, Y_train, vote_set, data_test)
    # result = dt_submit(X_train, Y_train, data_test)
    # result = bagging_submit(X_train, Y_train, data_test)
    # result = et_submit(X_train, Y_train, data_test)
    # result = rf_submit(X_train, Y_train, data_test)
    # result = ada_boosting_submit(X_train, Y_train, data_test)
    # result = gb_submit(X_train, Y_train, data_test)
    csv(p_id, result)



