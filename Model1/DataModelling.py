# This file is for the Kaggle Competition Titanic
# This file is to perform the data modelling
# Author: Kenny Hou
# Created at EY office desk 29.037C, 29/08/2019 3:14 PM

# Data analysis and wrangling
import pandas as pd

# Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import naive_bayes, gaussian_process, linear_model, neighbors, svm

# Acquire data
train_df = pd.read_csv('train_wrangled.csv')
test_df = pd.read_csv('test_wrangled.csv')
# combine = [train_df, test_df]

# Prepare training and testing data
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
# print(X_train.shape, Y_train.shape, X_test.shape)

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

# 1. Logistic Regression
def LogReg(Xtrain, Ytrain, Xtest):
    logreg = linear_model.LogisticRegression()
    logreg.fit(Xtrain, Ytrain)
    Y_pred = logreg.predict(Xtest)
    acc_log = round(logreg.score(Xtrain, Ytrain) * 100, 2)
    print(acc_log)  # 79.24

    # Check feature correlation
    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
    print(coeff_df.sort_values(by='Correlation', ascending=False))
    return Y_pred


# 2. Support Vector Machines
def SuVeMa(Xtrain, Ytrain, Xtest):
    svc = svm.SVC()
    svc.fit(Xtrain, Ytrain)
    Y_pred = svc.predict(Xtest)
    acc_svc = round(svc.score(Xtrain, Ytrain) * 100, 2)
    print(acc_svc)  # 95.17
    return Y_pred


# 3. KNN
def Knn(Xtrain, Ytrain, Xtest):
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(Xtrain, Ytrain)
    Y_pred = knn.predict(Xtest)
    acc_knn = round(knn.score(Xtrain, Ytrain) * 100, 2)
    print(acc_knn)  # 79.91
    return Y_pred


# 4. Gaussian Naive Bayes
def Gnb(Xtrain, Ytrain, Xtest):
    gaussian = linear_model.GaussianNB()
    gaussian.fit(Xtrain, Ytrain)
    Y_pred = gaussian.predict(Xtest)
    acc_gaussian = round(gaussian.score(Xtrain, Ytrain) * 100, 2)
    print(acc_gaussian)  # 75.98
    return Y_pred


# 5. Perceptron
def perc(Xtrain, Ytrain, Xtest):
    perceptron = linear_model.Perceptron()
    perceptron.fit(Xtrain, Ytrain)
    Y_pred = perceptron.predict(Xtest)
    acc_perceptron = round(perceptron.score(Xtrain, Ytrain) * 100, 2)
    print(acc_perceptron)  # 38.61
    return Y_pred


# 6. Linear SVC
def linearSVC(Xtrain, Ytrain, Xtest):
    linear_svc = linear_model.LinearSVC()
    linear_svc.fit(Xtrain, Ytrain)
    Y_pred = linear_svc.predict(Xtest)
    acc_linear_svc = round(linear_svc.score(Xtrain, Ytrain) * 100, 2)
    print(acc_linear_svc)  # 53.98
    return Y_pred


# 7. Stochastic Gradient Descent
def Sgd(Xtrain, Ytrain, Xtest):
    sgd = naive_bayes.SGDClassifier()
    sgd.fit(Xtrain, Ytrain)
    Y_pred = sgd.predict(Xtest)
    acc_sgd = round(sgd.score(Xtrain, Ytrain) * 100, 2)
    print(acc_sgd)  # 61.95
    return Y_pred


# 8. Decision Tree
def DeTree(Xtrain, Ytrain, Xtest):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(Xtrain, Ytrain)
    Y_pred = decision_tree.predict(Xtest)
    acc_decision_tree = round(decision_tree.score(Xtrain, Ytrain) * 100, 2)
    print(acc_decision_tree)  # 100
    return Y_pred


# 8. Random Forest
def RaForest(Xtrain, Ytrain, Xtest):
    random_forest = ensemble.RandomForestClassifier()
    random_forest.fit(Xtrain, Ytrain)
    Y_pred = random_forest.predict(Xtest)
    acc_random_forest = round(random_forest.score(Xtrain, Ytrain) * 100, 2)
    print(acc_random_forest)  # 98.65
    return Y_pred


# 9. Hard vote
def HardVote(Xtrain, Ytrain, Xtest):
    grid_hard = ensemble.VotingClassifier(estimators=vote_set, voting='hard')
    grid_hard.fit(Xtrain, Ytrain)
    Y_pred = grid_hard.predict(Xtest)
    return Y_pred


# 10. Soft vote
def SoftVote(Xtrain, Ytrain, Xtest):
    grid_soft = ensemble.VotingClassifier(estimators=vote_set, voting='soft')
    grid_soft.fit(Xtrain, Ytrain)
    Y_pred = grid_soft.predict(Xtest)
    return Y_pred


def submit(p_id, predict):
    submission = pd.DataFrame({'PassengerId': p_id, 'Survived': predict})
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    # Y_pred = LogReg(X_train, Y_train, X_test)
    # Y_pred = SuVeMa(X_train, Y_train, X_test)
    # Y_pred = Knn(X_train, Y_train, X_test)
    # Y_pred = Gnb(X_train, Y_train, X_test)
    # Y_pred = perc(X_train, Y_train, X_test)
    # Y_pred = linearSVC(X_train, Y_train, X_test)
    # Y_pred = Sgd(X_train, Y_train, X_test)
    # Y_pred = DeTree(X_train, Y_train, X_test)
    # Y_pred = RaForest(X_train, Y_train, X_test)  # 0.77990
    # Y_pred = HardVote(X_train, Y_train, X_test)  # 0.78947
    Y_pred = SoftVote(X_train, Y_train, X_test)  # 0.78947

    submit(test_df['PassengerId'], Y_pred)






