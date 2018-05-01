from preprocess import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod):
    """
    This function is for multi-class classification with some sampling methods.

    :param X: numpy array for features.
    :param Y: numpy array for class labels.
    :param sub_to_main_type: dict mapping network sub-type to network type.
    :param feature_names: a list of feature names.
    :param isSubType: flag for if labels in Y are network subtypes or not.
    :param samplingMethod: name of the sampling method. Valid names are: RandomOver, RandomUnder, SMOTE and None
    :return:
     cm: confusion matrix
     NetworkTypeLabels: a list of string, either network type or network subtype.
     accuracy: accuracy value taking a value in the range [0-1].
     feature_importances: a list of tuple of a feature's name and its importance in the classification.
    """

    if isSubType:
        NetworkTypeLabels = sorted(list(set(Y)), key=lambda sub_type: sub_to_main_type[sub_type])
    else:
        NetworkTypeLabels = sorted(list(set(Y)))

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.4, random_state=0)

    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

    if samplingMethod == "RandomOver":
        random_over = RandomOverSampler()
        sampled_x, sampled_y = random_over.fit_sample(X_train, y_train)

    elif samplingMethod == "RandomUnder":
        random_under = RandomUnderSampler()
        sampled_x, sampled_y = random_under.fit_sample(X_train, y_train)

    # SMOTE does not support multi-class classification in imblearn library, so we populate minority classes
    # in binary classification setting. The resulting set should all have the same # of instances as the largest class.
    elif samplingMethod == "SMOTE":
        sm = SMOTE(kind='regular', k=3)
        sm.fit(X_train, y_train)

        # get the label of the largest class in terms of the number of instances.
        majority = sm.maj_c_

        all_X = []
        all_Y = []

        for network_type in NetworkTypeLabels:
            if network_type != majority:
                # extract elements of a pair of network types, i.e. the majority and one to be inflated
                X_extracted = np.concatenate((X_train[y_train == majority], X_train[y_train == network_type]), axis=0)
                Y_extracted = np.concatenate((y_train[y_train == majority], y_train[y_train == network_type]), axis=0)
                x_tmp, y_tmp = sm.fit_sample(X_extracted, Y_extracted)
                x = x_tmp[y_tmp == network_type]
                y = y_tmp[y_tmp == network_type]
                all_X.append(x)
                all_Y.append(y)

        all_X.append(X_train[y_train == majority])
        all_Y.append(y_train[y_train == majority])

        Xs = np.concatenate(tuple(all_X))
        Ys = np.concatenate(tuple(all_Y))

        sampled_x, sampled_y = sm.fit_sample(Xs, Ys)

    elif samplingMethod == "None":
        sampled_x, sampled_y = X_train, y_train

    random_forest = RandomForestClassifier()
    random_forest.fit(sampled_x, sampled_y)
    accuracy = random_forest.score(X_test, y_test)

    feature_importances = sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_names),
                                 reverse=True)

    y_pred = random_forest.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=NetworkTypeLabels)
    return cm, NetworkTypeLabels, accuracy, feature_importances
