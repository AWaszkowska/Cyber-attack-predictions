# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('/home/amelia97/Documents/Python/aml/AML-2023-Project1/AML-2023-Project1/dataset/train_students.csv')
test = pd.read_csv('/home/amelia97/Documents/Python/aml/AML-2023-Project1/AML-2023-Project1/dataset/test_students.csv')
print(test.head())

# check for nan values

print(train.isnull().any().any())
print(test.isnull().any().any())

# check datatypes

print(train.dtypes)
print(test.dtypes)
types = test.dtypes

cols_to_transform = train.select_dtypes(include=['object']).columns
print(cols_to_transform)

# outliers check
# dummy label encode, just for the purpose of plotting boxplot
# before cutting outliers, check their association with target

# dictionary to store the inverse mapping for each column
inverse_mapping = {}


for col in cols_to_transform:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    inverse_mapping[col] = le.inverse_transform(np.arange(len(le.classes_)))
    
target = train['attack_type']

for col in train.columns:
    if train[col].dtype != 'object' and train[col].notna().all():
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr
        outliers = train[(train[col] < lower) | (train[col] > upper)][col]
        if not outliers.empty:
            # plt.figure()
            # plt.hist(train[col], bins=20)
            # plt.title(col)
            # plt.scatter(train[col], target)
            # plt.xlabel(col)
            # plt.ylabel('target')
            #plt.show()
            correlation = train[[col, 'attack_type']].corr().iloc[0,1]
            if correlation > 0.5:
               train = train.loc[~((train[col] < lower) | (train[col] > upper)), :]
               print(col)


# undo label encoding
for col in cols_to_transform:
    train[col] = inverse_mapping[col][train[col]]

# convert non-numerical data to numerical
# first we check how many uniqe vals, because if many, onehot may lead to sparcity
for col in train[cols_to_transform]:
    print(train[col].nunique())

# thecolumn 'service' has 69 unique instances, which might be too much for onehot, so we use feature hashing

hash_vect = HashingVectorizer(n_features=50)
train_service_hashed = hash_vect.transform(train['service'])
test_service_hashed = hash_vect.transform(test['service'])

# replace the original 'service' column with the hashed version in the train and test dataframes
train.drop('service', axis=1, inplace=True)
train = pd.concat([train, pd.DataFrame(train_service_hashed.toarray())], axis=1)

test.drop('service', axis=1, inplace=True)
test = pd.concat([test, pd.DataFrame(test_service_hashed.toarray())], axis=1)

# for the rest we can onehot

cols_to_onehot = ['protocol_type', 'flag']
encoder = OneHotEncoder(sparse = False)
train_cat = encoder.fit_transform(train[cols_to_onehot])
train_cat = pd.DataFrame(train_cat, columns=encoder.get_feature_names_out(cols_to_onehot))

test_cat = encoder.fit_transform(test[cols_to_onehot])
test_cat = pd.DataFrame(test_cat, columns=encoder.get_feature_names_out(cols_to_onehot))

# tg = ['attack_type']
# enc = OneHotEncoder(sparse = False)
# Y_train = enc.fit_transform(train[tg])
# Y_train = pd.DataFrame(Y_train, columns=enc.get_feature_names_out(tg))

mapping = {'normal': 0, 'Dos': 1, 'R2L': 2, 'U2R': 3, 'Probe': 4}
train['attack_type'] = train['attack_type'].map(mapping)
Y_train = pd.DataFrame(train['attack_type'], columns=['attack_type'])
print(Y_train.head())
# =============================================================================
# lnc = LabelEncoder()
# Y_train = lnc.fit_transform(train['attack_type'])
# print(lnc.classes_)
# labels = np.array(['Dos', 'Probe', 'R2L', 'U2R', 'normal'])
# =============================================================================


# one-hot over dummies, because OHE saves the exploded categories into 
# it’s object, if the total number of unique values in a cat column is 
# not the same for my train set vs test set, I’m going to have problems.

# normalize (not standarize, because data don't follow normal distrib) numerical data
numeric_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

norm = MinMaxScaler()
# transform testing data
X_test = norm.fit_transform(test[numeric_features])

norm = MinMaxScaler()
# transform training data
train_num = norm.fit_transform(train[numeric_features])

# concatenate numerical and cat
X_train = pd.concat([pd.DataFrame(train_cat), pd.DataFrame(train_num)], axis=1)
test = pd.concat([pd.DataFrame(test_cat), pd.DataFrame(X_test)], axis=1)

# pca
pca = PCA(n_components = 36)
X_train = pca.fit_transform(X_train)
# applying the same transformation to the testing set
test = pca.transform(test)
 
explained_variance = pca.explained_variance_ratio_

training_set = pd.concat([pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis=1)

#%% 

# Split the training dataset into training, validation, and test data
X = training_set.iloc[:, :-1]
y = training_set.iloc[:, -1:]

X_train_val, X_testing, y_train_val, y_testing = train_test_split(X, y, test_size=0.2, random_state=1)
X_training, X_val, y_training, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1)

# Reshape the target variable to a 1D array
y_training = y_training.values.ravel()
y_val = y_val.values.ravel()

#svm model
from sklearn import metrics
from sklearn.metrics import confusion_matrix


clf = SVC() 

# Define the range of hyperparameters to be tuned
params = {'C': [1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [1, 10]}
# grid search 
grd_search = GridSearchCV(estimator=clf, param_grid=params, cv=5)
grd_search.fit(X_training, y_training)

# Select the best model and train it on the validation set
best_svm = grd_search.best_estimator_
best_svm.fit(X_val, y_val)

# Evaluate the performance of the selected model on the test set
tst_score = best_svm.score(X_testing, y_testing)
print("Test set accuracy: {:.2f}".format(tst_score))

# =============================================================================
# clf.fit(X_training, y_training)
# clf.fit(X_train_val, y_train_val)
# pred = clf.predict(X_testing)
# accuracy = accuracy_score(y_testing, pred)
# =============================================================================


preds = clf.predict(X_testing)
conf_matrix = confusion_matrix(y_true=y_val, y_pred=preds)
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# random forest

# define hyperparameters and their possible values
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
    }
rfc = RandomForestClassifier()

# create a grid search object with the defined hyperparams
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
grid_search.fit(X_training, y_training)

# evaluate the performance of each model on validset
best = grid_search.best_estimator_
valscore = best.score(X_val, y_val)
print("Best model validation score: {:.2f}".format(valscore))

# Evaluate the performance of the selected model on the test set
# The score() method of the RandomForestClassifier class returns 
# the mean accuracy of the predictions for the given input data.
test_score = best.score(X_testing, y_testing)
print("Best model test score: {:.2f}".format(test_score))

# logistic regression
log_reg = LogisticRegression(max_iter=1000)

# Define hyperparameters to be tuned
p_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}
# Perform a grid search with cross-validation on the training set
g_search = GridSearchCV(estimator=log_reg, param_grid=p_grid, cv=5)
g_search.fit(X_training, y_training)

# Evaluate the performance of each model on the validation set
for i, model in enumerate(g_search.cv_results_['params']):
    print("Model {}: {}".format(i, model))
    print("Validation score: {:.2f}".format(g_search.cv_results_['mean_test_score'][i]))

# Select the best model and train it on the combined training and validation sets
best_log_reg = g_search.best_estimator_
best_log_reg.fit(X_train_val, y_train_val)

# Evaluate the trained model on the test set
test_score = best_log_reg.score(X_testing, y_testing)
print("Test set accuracy: {:.2f}".format(test_score))

# test set

df_test_final = test
df2 = pd.DataFrame()
A = df_test_final
A_predictions = best.predict(A)
df2['SampleID'] = range(0, len(A_predictions))
df2['Class'] = A_predictions
df2.to_csv('submission.csv',index = False)






