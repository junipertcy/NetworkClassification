import numpy as np
from preprocess import init
from plot import plot_feature_importance, plot_2d
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def convert_one_to_many(X, Y, label):
    """
    Convert multi-class labels into binary class labels. The argument label corresponds to the label of interest.
    """
    X_converted = []
    Y_converted = []

    for x, y in zip(X, Y):
        X_converted.append(x)
        if y == label:
            Y_converted.append(1)
        else:
            Y_converted.append(0)

    return X_converted, Y_converted


def split_train_test(X, Y):
    """
    Split the data set into training and test sets.
    """
    X = np.array(X)
    Y = np.array(Y)

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.4, random_state=0)
    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

    return X_train, X_test, y_train, y_test


def one_to_many_classification(X_train, X_test, y_train, y_test, feature_order):
    """
    Binary classification of one kind vs. others (many).
    """

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    accuracy = random_forest.score(X_test, y_test)
    y_pred = random_forest.predict(X_test)
    feature_importance = sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_order),
                                reverse=True)
    AUC = roc_auc_score(y_test, y_pred)

    return accuracy, feature_importance, AUC


def many_classifications(X, Y, feature_order, N):
    """
    Does one_to_many_classification N times and aggregate the outputs from it.
    """
    list_important_features = []
    list_accuracies = []
    list_auc = []
    for i in range(N):
        print
        "i:%d" % i
        X_train, X_test, y_train, y_test = split_train_test(X, Y)
        accuracy, feature_importances, auc = one_to_many_classification(X_train, X_test, y_train, y_test, feature_order)
        list_important_features.append(feature_importances)
        list_accuracies.append(accuracy)
        list_auc.append(auc)

    return list_accuracies, list_important_features, list_auc


def main():
    column_names = ["NetworkType", "SubType", "ClusteringCoefficient", "DegreeAssortativity", "m4_1", "m4_2", "m4_3",
                    "m4_4", "m4_5", "m4_6"]

    isSubType = True
    at_least = 1
    X, Y, sub_to_main_type, feature_order = init("features.csv", column_names, isSubType, at_least)
    N = 100

    # network subtype one is interested in
    one = "Communication"

    X_converted, Y_converted = convert_one_to_many(X, Y, one)
    list_accuracies, list_important_features, list_auc = many_classifications(X_converted, Y_converted,
                                                                              sub_to_main_type, feature_order, N)

    print
    "average accuracy: %f" % (float(sum(list_accuracies)) / float(N))
    print
    "average AUC: %f" % (float(sum(list_auc)) / float(N))

    dominant_features = plot_feature_importance(list_important_features, feature_order)

    first = dominant_features[0][0][0]
    second = dominant_features[1][0][0]
    if first == second:
        second = dominant_features[1][1][0]
    Y_converted_string_labels = [one if y == 1 else "Other" for y in Y_converted]
    print
    "first: ", first
    print
    "second: ", second

    x_label = first
    y_label = second
    x_index = feature_order.index(x_label)
    y_index = feature_order.index(y_label)
    plot_2d(np.array(X_converted), np.array(Y_converted_string_labels), x_index, y_index, x_label, y_label)


if __name__ == '__main__':
    main()
