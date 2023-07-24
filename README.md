# Cyber-attack-predictions
A ML model for predicting a type of cyber attack
1. Work objective and dataset description

The objective of this project is to explore, analyze and interpret the findings in the provided dataset and develop a model, through the use of preprocessing techniques and machine learning algorithms, that would allow us to predict if the system was attacked, and if it was, what type of attack.
Two datasets, one to train, validate and test the models (train_students.csv), and a dataset to predict if the system was being attacked, and if it was, what type of attack, and submit the results to the Kaggle competition (test_students.csv). The datasets consist of 103961 x 42 columns and 44556 rows x 42 columns respectively. Both datasets had the same features for 41 columns, the difference in columns is that the train_students.csv dataset had the attack_type column and the test_students.csv dataset had the SampleID column, which is an index column.
The train_students.csv dataset had four categorical features (service, attack_type, flag, 
The train_students.csv dataset had four categorical features (service, attack_type, flag, protocol_type) and 38 numeric features. The test_students.csv dataset had three categorical features (service, protocol_type, flag) and 39 numerical features.


2. Feature Engineering

a) checking for missing values 
b) checking datatypes and creating a list of columns that need transformation
c) label encoding only for the purpose of the outlier check
d)  outliers check
    Check in a loop whether a datapoint is an outlier or not, according to the interquartile rule. Check whether a correlation between the datapoint and the target is greater than 0.5 and if so, the datapoint shall be removed.
e) undoing label encoding
f) checking how many unique values there is in each of the categorical features
g) hash encoding the column "service"

Feature hashing - this is a method for turning arbitrary features into a sparse binary vector. It does not require pre-built dictionary. It is a way to deal with categorical features with lots of unique values, avoiding curse of dimensionality created by one-hot encoding. A large set of input features is mapped to a fixed-size feature vector using a hash function. This process involves mapping each input feature to a hash value, which is then used as an index into a fixed-size array or vector.

h) one-hot encoding the rest of categorical features
i) label encoding the target using .map function (previously, used usual label encoding but ended up with different ctegory-to-label encodeing than in the paper provided)
j) normalizing numerical features using MinMaxScaler, because the data don't follow normal distribution (so using standardization is less preferable)
k) Principal Component Analysis


3. Model training

Trained our models with data splitted into three parts: training, validation and testing set. The validation set is used to evaluate the performance of the model during training and to tune the hyperparameters. The typical split ratio is 80:20 or 90:10.

Then, used automatic hyperparameter tuning technique called grid search, in order to select the best hyperparameters possible.

Hyperparameter tuning

There are different techniques for tuning hyperparameters, main of them are: grid search, random search, or Bayesian optimization. Grid search is a simple and straightforward technique, which evaluates all possible combinations of hyperparameters in a predefined range. Random search is similar, but instead of evaluating all possible combinations, it randomly samples a subset of the search space. Bayesian optimization is a more advanced technique using a probabilistic model to predict the performance of different hyperparameter configurations, and selecting the next configuration to evaluate based on the predicted performance.

Logistic regression

Initially, got a following error: 
/home/amelia97/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

That indicates that the logistic regression algorithm did not converge during aptimization , which means it did not find the optimal solution. This is quite common issue in logistic regression and it can be caused by following promlems:

- the learning rate is too high and the optimal solution was overshot
- the model is too complex, for example has too many features
- the data is not well-suited for logistic regression, the relationship between features and target is not linear

Then, increased the number of iterations to 1000 and the algorithm converged.

The most important hyperparameters are:

- solver - this is the algorithm for the optimization problem. Available are: Newton-CG (uses the Newton's method to minimize the cost function, which requires computing the hessian matrix, which can be computationally expensive for large datasets), LBFGS (uses limited- memory Broyden–Fletcher–Goldfarb–Shanno algorithm to minimize the cost function. It's a quasi-Newton method that approximates the Hessian matrix using information from the previous iterations), liblinear (uses coordinate gradient descent, is said to be efficient for highh-dimensional data), sag (uses the stochastic average gradient descent), saga (extension to sag, uses an adaptive regularization parameter).

- penalty -  is supposed to reduce model generalization error and regulate overfitting. The choices are: l1, l2, elasticnet and none. Some penalties may not work with some solvers.

- C - regularization strength, a positive float. Smaller values specify stronger regularization and high value tells the model to give high weight to the training data.

- max_iter - maximum number of iteration, may help with achieving convergence.

Random Forest

The most important hyperparameters are:

- max_depth - controls the maximum number of levels or nodes allowed in each decision tree in the forest. The deeper the tree, the more complex the model, so setting a max depth can prevent overfitting and improve model's ability to generalize. A decision node leads to other nodes. It represents a question to be answeared, e.g. "Will we go on a trip?". A leaf node is a terminal node representing a decision, e.g. "We will go on a trip."

- min_sample_split - minimum number of samples required to split an internal node (default: 2)

- max_leaf_nodes - maximum number of leaf nodes a tree can have

-  min_samples_leaf - minimum number of samples required to be at a leaf node (default:1)

- n_estimators - number of trees in the forest. A tree is one of many models that are combined to create the ensemble model. Each tree is constructed using a random subset of the training dataset and a random subset of features at each decision node.

- max_sample - determines the fraction of the original dataset that is given to any individual tree

- max_features - the number of features to consider when looking for the best split

Best result obtained for 'none' max_depth.

Support Vector Machines

The most important hyperparameters are:

- C - the penalty parameter, responsible for trade-off between maximizing the margin and minimizing the classification error. Smaller C means more margin violations and a simpler model.

- kernel - function used to transform the data into a higher-dimensional space where a linear decision boundary can be found

- gamma - kernel coefficient. Smaller gamma means a broader kernel and a smoother decision boundary.

- degree - The degree of the polynomial kernel function.

- class_weight - A parameter that assigns different weights to different classes to address class imbalance issues.

Best results for:
Model 5: {'C': 10, 'penalty': 'l2'}
Validation score: 0.94
