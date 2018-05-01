from preprocess import init
import matplotlib.pyplot as plt


def separator(X, Y, c1, c2):
    """
    Separates c1 and c2 from the data.
    """
    c1_X = []
    c1_Y = []
    c2_X = []
    c2_Y = []

    for x, y in zip(X, Y):
        if y == c1:
            c1_X.append(x)
            c1_Y.append(y)
        elif y == c2:
            c2_X.append(x)
            c2_Y.append(y)

    return c1_X, c1_Y, c2_X, c2_Y


def base_to_predict(base_X, base_Y, predict_X, predict_Y):
    """
    Train on the base networks, classify predict networks
    """
    random_forest = RandomForestClassifier()
    random_forest.fit(base_X, base_Y)
    y_pred = random_forest.predict(predict_X)
    return zip(y_pred, predict_Y)


def plot(c1_X, c2_X, feature_order, c1_name, c2_name):
    plt.plot(range(len(c1_X[0])), c1_X[0], color="r", label=c1_name)
    for c1_features in c1_X[1:]:
        plt.plot(range(len(c1_features)), c1_features, color="r")

    plt.plot(range(len(c2_X[0])), c2_X[0], color="b", label=c2_name)
    for c2_features in c2_X[1:]:
        plt.plot(range(len(c2_features)), c2_features, color="b")

    plt.xticks(range(len(c1_features)), feature_order, rotation='vertical')
    plt.legend()
    plt.show()


def main():
    column_names = ["NetworkType", "SubType", "ClusteringCoefficient", "DegreeAssortativity", "m4_1", "m4_2", "m4_3",
                    "m4_4", "m4_5", "m4_6"]

    isSubType = True
    at_least = 1
    X, Y, sub_to_main_type, feature_order = init("features.csv", column_names, isSubType, at_least)
    N = 100

    c1_name = "ER Network"
    c2_name = "PeerToPeer"
    # c2 to c1
    c1_X, c1_Y, c2_X, c2_Y = separator(X, Y, c1_name, c2_name)
    plot(c1_X, c2_X, feature_order, c1_name, c2_name)


if __name__ == '__main__':
    main()
