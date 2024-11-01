import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from itertools import product


def train_logreg(
        X_train, y_train, X_valid, y_valid,
        Cs=[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]):
    '''
    Train a logistic regression by cross-validation.

    Returns
    -------
    best_logreg
        A LogisticRegression object.
    '''
    best_valid_auc = -1
    for C in Cs:
        logreg = Pipeline(
                [('scaler', preprocessing.StandardScaler()),
                 ('logReg', LogisticRegression(
                    C=C, solver='lbfgs',
                    multi_class='auto',
                    random_state=0, max_iter=1000))])

        logreg.fit(X_train, y_train)
        valid_pred = logreg.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_logreg = logreg

    # Refit with best parameters
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_logreg = best_logreg.fit(X, y)

    return best_logreg


def train_linear_reg(X_train, y_train, X_valid, y_valid):
    '''
    Train a linear regression by cross-validation.

    Returns
    -------
    best_linear_reg
        A Ridge object.
    '''
    best_valid_mse = float('inf')
    for alpha in [0.01, 0.1, 1, 10, 100]:
        linear_reg = Pipeline(
                [('scaler', preprocessing.StandardScaler()),
                 ('ridge', Ridge(
                     alpha=alpha, random_state=0, max_iter=1000))])

        linear_reg.fit(X_train, y_train)
        valid_pred = linear_reg.predict(X_valid)
        valid_mse = np.sum(np.square(y_valid - valid_pred))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_linear_reg = linear_reg

    # Refit with best parameters
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_linear_reg = best_linear_reg.fit(X, y)

    return best_linear_reg


def train_decision_tree(X_train, y_train, X_valid, y_valid,
                        min_samples_leaf_options=[10, 25, 100]):
    '''
    Train a decision tree classifier by cross-validation.

    Returns
    -------
    best_dectree
        A DecisionTreeClassifier object.
    '''
    assert len(y_train.shape) == 1
    best_valid_auc = -1
    # min_samples_leaf_options = [10, 25, 100]
    if X_train.shape[0] < 10:
        min_samples_leaf_options = [1]
    for min_samples_leaf in min_samples_leaf_options:
        if min_samples_leaf > X_train.shape[0]:
            continue
        dectree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf,
                random_state=0)
        dectree.fit(X_train, y_train)
        valid_pred = dectree.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_dectree = dectree

    # Refit with best parameters
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_dectree = best_dectree.fit(X, y)

    return best_dectree


def train_decision_tree_reg(
        X_train, y_train, X_valid, y_valid, 
        min_samples_leaf_options=[10, 25, 100], 
        max_depth_options=None, min_samples_leaf_default=None):
    '''
    Train a decision tree regressor by cross-validation.

    Returns
    -------
    best_dectree
        A DecisionTreeRegressor object.
    '''
    assert min_samples_leaf_options is not None or max_depth_options is not None
    best_valid_mse = float('inf')
    if min_samples_leaf_options is not None:
        if X_train.shape[0] < 10:
            min_samples_leaf_options = [1]
        for min_samples_leaf in min_samples_leaf_options:
            if min_samples_leaf > X_train.shape[0]:
                continue
            dectree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=0)
            dectree.fit(X_train, y_train)
            valid_pred = dectree.predict(X_valid)
            valid_mse = np.sum(np.square(y_valid - valid_pred))
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_dectree = dectree
    else:
        for max_depth in max_depth_options:
            if min_samples_leaf_default:
                dectree = DecisionTreeRegressor(
                        max_depth=max_depth, 
                        min_samples_leaf=min_samples_leaf_default,
                        random_state=0)
            else:
                dectree = DecisionTreeRegressor(
                        max_depth=max_depth,
                        random_state=0)
            dectree.fit(X_train, y_train)
            valid_pred = dectree.predict(X_valid)
            valid_mse = np.sum(np.square(y_valid - valid_pred))
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_dectree = dectree

    # Refit on all the data
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_dectree = best_dectree.fit(X, y)

    return best_dectree


def train_random_forest(X_train, y_train, X_valid, y_valid, min_samples_leaf_options = [10, 25, 100]):
    '''
    Train a random forest classifier by cross-validation.

    Returns
    -------
    best_forest
        A RandomForestClassifier object.
    '''
    assert len(y_train.shape) == 1
    best_valid_auc = -1
    #min_samples_leaf_options = [10, 25, 100]
    if X_train.shape[0] < 10:
        min_samples_leaf_options = [1]
    for n_trees, min_samples_leaf in product([10, 25, 100], min_samples_leaf_options):
        if min_samples_leaf > X_train.shape[0]:
            continue
        forest = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=0)
        forest.fit(X_train, y_train)
        valid_pred = forest.predict_proba(X_valid)[:,1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_forest = forest

    # Refit on all the data
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_forest = best_forest.fit(X, y)

    return best_forest


def train_random_forest_reg(X_train, y_train, X_valid, y_valid, min_samples_leaf_options = [10, 25, 100]):
    '''
    Train a random forest regressor by cross-validation.

    Returns
    -------
    best_forest
        A RandomForestRegressor object.
    '''
    best_valid_mse = float('inf')
    if X_train.shape[0] < 10:
        min_samples_leaf_options = [1]
    for n_trees, min_samples_leaf in product([10, 25, 100], min_samples_leaf_options):
        if min_samples_leaf > X_train.shape[0]:
            continue
        forest = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=0)
        forest.fit(X_train, y_train)
        valid_pred = forest.predict(X_valid)
        valid_mse = np.sum(np.square(y_valid - valid_pred))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_forest = forest

    # Refit on all the data
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_forest = best_forest.fit(X, y)

    return best_forest