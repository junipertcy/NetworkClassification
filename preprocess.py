import csv
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer


def data_read(filepath, *features, **kwargs):
    """
    Read a csv file (features.csv) and returns a dictionary
    (key ->  gml file name; value -> dict of features whose keys are feature name and
     values are value of the feature).
    """
    network_dict = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:

            filtered = dict((k, v) for k, v in row.items() if all([(k in features), v, (v != "nan")]))
            # print("filtered = ", filtered)
            # if filtered lacks some features, e.g. not calculated yet.
            if len(filtered) != len(features):
                continue

            # you can skip a row based on the NetworkType
            elif row["NetworkType"] in kwargs.get("exclusive_types", []):
                continue

            # below is for extracting only specific kinds of networks
            elif row["NetworkType"] in kwargs.get("inclusive_types", []):
                gml_name = row[".gmlFile"]
                network_dict[gml_name] = filtered

            else:
                gml_name = row[".gmlFile"]
                # print('gml_name = ', gml_name)
                network_dict[gml_name] = filtered

    return network_dict


def XY_generator(network_dict):
    """
    Separate labels and features values for scikit-learn algorithms.
    """

    X = []
    Y = []

    for gml_name in network_dict:
        d = network_dict[gml_name]  # d for dictionary.
        Y.append((gml_name, d["NetworkType"], d["SubType"]))
        X.append(dict((k, v) for k, v in d.items() if not (k == "NetworkType" or k == "SubType")))

    return X, Y


def filter_float(network_dict):
    """
    Make a string of float in the dictionary into float.
    """
    for gml_name in network_dict:
        for e in network_dict[gml_name]:
            if not ((e == "NetworkType") or (e == "SubType")):
                network_dict[gml_name][e] = float(network_dict[gml_name][e])
    return network_dict


def XY_filter_unpopular(X, Y, threshold):
    """
    filters out the elements which are unpopular (i.e. # of ys is below threshold).
    """
    counts = Counter(Y)
    popular = [item[0] for item in counts.most_common() if item[1] > threshold]

    # popular = []  # TODO: ad-hoc fix
    # popular = [elem for (elem, count) in filter(lambda (e, c): c > threshold, counts.most_common())]
    return np.concatenate(tuple(X[Y == p] for p in popular), axis=0), \
           np.concatenate(tuple(Y[Y == p] for p in popular), axis=0)


def init(filepath, column_names, isSubType, at_least, **kwargs):
    """

    Parameters
    ----------
    filepath:
        path to a file containing feature values (e.g. features.csv).
    column_names:
        names of columns in the csv file to be read. E.g. ["NetworkType","SubType","Modularity",...]
    isSubType:
        flag for if labels in Y are network subtypes or not.
    at_least:
        integer threshold for filtering the minority classes below the number
    kwargs

    Returns
    -------
    X:
        numpy array for features.
    Y:
        numpy array for class labels.
    sub_to_main_type:
        dict mapping network sub-type to network type.
    feature_order:
        a list of strings (features' names) ordered for the feature array X.
        That is, columns of X correspond to feature_order.

    """
    network_dict = data_read(filepath, *column_names, **kwargs)

    network_dict = filter_float(network_dict)
    # print("network_dict = ", network_dict)
    features, labels = XY_generator(network_dict)

    v = DictVectorizer(sparse=False)
    # print("features = ", features)
    X = v.fit_transform(features)

    feature_order = map(lambda x: x[0], sorted(v.vocabulary_.items(), key=lambda x: x[1]))

    sub_to_main_type = dict((SubType, NetworkType) for gml, NetworkType, SubType in labels)
    print("IMPORTANT labels: {}".format(labels))
    if isSubType:
        Y = np.array([SubType for gml, NetworkType, SubType in labels])
    else:
        Y = np.array([NetworkType for gml, NetworkType, SubType in labels])

    X, Y = XY_filter_unpopular(X, Y, at_least)

    return X, Y, sub_to_main_type, list(feature_order)
