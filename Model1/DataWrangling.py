# This file is for the Kaggle Competition Titanic
# This file is to perform the data wrangling
# Author: Kenny Hou
# Created at EY office desk 29.033E, 28/08/2019 2:28 PM

# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Acquire data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# combine = [train_df, test_df]

# Dropping unrelated features
# print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combination = [train_df, test_df]
# print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# Creating new feature extracting from existing
# 1. Create Title feature from name feature
def create_title(train, test, combine):
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    # pd.crosstab(train_df['Title'], train_df['Sex'])

    # Replace rare titles with common name or classify them as Rare
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                     'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

    # Convert categorical titles to ordinal
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs:": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset["Title"].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    # pd.set_option('display.max_columns', 100)  # display full columns
    # print(train_df.head())

    # Drop the Name and PassengerID features
    train_df = train.drop(['Name', 'PassengerId'], axis=1)
    test_df = test.drop(['Name'], axis=1)
    combination = [train_df, test_df]
    # print(train_df.shape, test_df.shape)

    return train_df, test_df, combination


# 2. Convert categorical sex feature from strings to numerical values
def sex_to_num(combine):
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    # print(train_df.head())
    return combine


# 3. Complete numerical continuous age feature: fix feature with missing or null values
'''
We can consider three methods to complete a numerical continuous feature.

1. A simple way is to generate random numbers between mean and standard deviation.
2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation
among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature
combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and
standard deviation, based on sets of Pclass and Gender combinations.

Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will
prefer method 2.
'''
def age_fix(train, combine):
    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
    # grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
    # grid.add_legend()

    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # convert random age float to nearest .5 age
                guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    # Create age bands and determine correlations with survived
    train['AgeBand'] = pd.cut(train['Age'], 5)
    # print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False)
    #       .mean().sort_values(by='AgeBand', ascending=True))

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    train_df = train.drop(['AgeBand'], axis=1)
    combination = [train_df, test_df]
    return train_df, combination


# 4. Create new feature combining existing features
def is_along(train, test, combine):
    # Create feature FamilySize
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False)
    #       .mean().sort_values(by='Survived', ascending=False))

    # Create feature IsAlone
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False)).mean()

    # Drop Parch, SibSp and FamilySize features in favor of IsAlone
    train_df = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    # Create an artificial feature combining Pclass and Age
    for dataset in combine:
        dataset['AgeClass'] = dataset.Age * dataset.Pclass
    # print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
    return train_df, test_df, combine


# 5. Complete Embarked categorical feature and convert it to numeric
'''
Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values.
We simply fill these with the most common occurance.
'''
def embark_to_num(train, combine):
    freq_port = train.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    # print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False)
    #       .mean().sort_values(by='Survived', ascending=False))

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    return combine


# 6. Quickly complete and convert numeric Fare feature
'''
We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs
most frequently for this feature. We do this in a single line of code.

Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing
feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to
operate on non-null values.

We may also want round off the fare to two decimals as it represents currency.
'''
def fare(train, combine):
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    # create FareBand feature
    train['FareBand'] = pd.qcut(train['Fare'], 4)
    # print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False)
    #       .mean().sort_values(by='FareBand', ascending=True))

    # Convert Fare feature to ordinal values based on the FareBand
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.9, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.9) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train.drop(['FareBand'], axis=1)
    return train_df


def write_csv(train, test):
    train.to_csv('train_wrangled.csv')
    test.to_csv('test_wrangled.csv')


if __name__ == '__main__':
    train_df, test_df, combination = create_title(train_df, test_df, combination)
    combination = sex_to_num(combination)
    train_df, combination = age_fix(train_df, combination)
    train_df, test_df, combination = is_along(train_df, test_df, combination)
    combination = embark_to_num(train_df, combination)
    train_df = fare(train_df, combination)
    write_csv(train_df, test_df)



