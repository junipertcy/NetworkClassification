import math
import numpy as np
from preprocess import init
from multiclass import multiclass_classification
from plot import plot_confusion_matrix
from plot import plot_feature_importance
from plot import index_to_color
from plot import MDS_plot
import matplotlib.pyplot as plt
# import pylab
import scipy.cluster.hierarchy as sch
import networkx as nx

import click


colors_domain = ["#ff0000", "#9c8110", "#00d404", "#00a4d4", "#1d00d4", "#a400c3", "#831e1e"]


def sum_confusion_matrix(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod, N):
    accum_matrix, NetworkTypeLabels, accum_acc, feature_importances = \
        multiclass_classification(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod)

    list_important_features = [feature_importances]

    for i in range(N - 1):
        print("i: ", i)
        cm, _, accuracy, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_order,
                                                                         isSubType, samplingMethod)
        accum_matrix += cm
        accum_acc += accuracy
        list_important_features.append(feature_importances)
    return accum_matrix, NetworkTypeLabels, accum_acc, list_important_features


def make_symmetric(cm):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_filtered = list(map(lambda ax: list(map(lambda val: 0.0 if math.isnan(val) else val, ax)), cm_normalized))
    N = len(cm_normalized_filtered)

    # make cm symmetric
    for i in range(N):
        for j in range(N):
            if i == j:
                cm_normalized_filtered[i][j] = 0
            else:
                maximum = max([cm_normalized_filtered[i][j], cm_normalized_filtered[j][i]])
                cm_normalized_filtered[i][j] = maximum
                cm_normalized_filtered[j][i] = maximum

    # make values into distance
    for i in range(N):
        for j in range(N):
            if i == j: continue
            cm_normalized_filtered[i][j] = (1 - cm_normalized_filtered[i][j]) * 100

    return np.asarray(cm_normalized_filtered)


def build_dendrogram(D, leave_name, sub_to_main_type, isSubType):
    Domains = list(set(sub_to_main_type.values()))
    color_map = index_to_color(Domains, "jet")
    fig = plt.figure(figsize=(10, 10))
    Y = sch.linkage(D, method='complete')  # , method='centroid')
    Z1 = sch.dendrogram(Y, orientation='right', labels=leave_name)
    ax = plt.gca()
    ylbls = ax.get_ymajorticklabels()

    if isSubType:
        for lbl in ylbls:
            domain = sub_to_main_type[lbl.get_text()]
            index = Domains.index(domain)
            lbl.set_color(color_map(index))

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    fig.show()
    fig.savefig('only_dendrogram_.png', bbox_inches='tight')


def min_or_max(G, func=max):
    return func([attr["weight"] for i, j, attr in G.edges(data=True)])


def threshold(G, alpha):
    thresholded_graph = nx.Graph()

    for u, v, w, in G.edges(data='weight'):
        if w > alpha:
            thresholded_graph.add_edge(u, v, weight=w * 50)

    return thresholded_graph


def graph_draw(G, NetworkTypeLabels, sub_to_main_type):
    G = threshold(G, 0.00)
    pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.spring_layout(G)
    labels = {}
    for e in G.nodes():
        labels[e] = NetworkTypeLabels[e]

    Domains = list(set(sub_to_main_type.values()))

    # color_map = index_to_color(Domains,"hsv")
    color_map = lambda i: colors_domain[i]
    print(NetworkTypeLabels)
    print(sub_to_main_type)
    colors = [color_map(Domains.index(sub_to_main_type[sub_domain])) for sub_domain in NetworkTypeLabels]

    minimum = min_or_max(G, min)  # the minimum of weights
    maximum = min_or_max(G, max)  # the maximum of weights
    n = maximum - minimum

    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=11)
    edge_alpha = map(lambda x: round(x, 4), np.linspace(0.25, 0.8, n))

    for e, v, w in list(G.edges(data='weight')):
        print("e,v,w:", (e, v, w))
        nx.draw_networkx_edges(G, pos=pos, edgelist=[(e, v)], alpha=0.6, width=w * 0.2)

    nx.draw_networkx_nodes(G, pos=pos, nodelist=G.nodes(), node_size=250, node_color=colors, alpha=0.6)

    plt.axis('off')
    plt.show()


def make_adj_matrix(cm):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_filtered = list(map(lambda ax: list(map(lambda val: 0.0 if math.isnan(val) else val, ax)), cm_normalized))
    N = len(cm_normalized_filtered)

    # make cm symmetric
    for i in range(N):
        for j in range(N):
            if i == j:
                cm_normalized_filtered[i][j] = 0
            else:
                maximum = max([cm_normalized_filtered[i][j], cm_normalized_filtered[j][i]])
                cm_normalized_filtered[i][j] = maximum
                cm_normalized_filtered[j][i] = maximum

    return np.asarray(cm_normalized_filtered)


@click.command()
@click.option('--csv', nargs=1, type=str, help='CSV data for the features.')
@click.option('--feature', '-f', multiple=True)
def main(csv, feature):
    # The order in this list should be the same as columns in features.csv
    # column_names = ["NetworkType", "SubType", "ClusteringCoefficient", "DegreeAssortativity",
    #                 "m4_1", "m4_2", "m4_3", "m4_4", "m4_5", "m4_6"]
    # features: "sepal_length", "sepal_width", "petal_length", "petal_width"
    column_names = ["NetworkType", "SubType"]
    column_names += list(feature)
    print(column_names)
    isSubType = True

    csv_file = csv

    # at_least is used for filtering out classes whose instance is below this threshold.
    at_least = 6
    X, Y, sub_to_main_type, feature_order = init(csv_file, column_names, isSubType, at_least)

    # the number of iteration for multi-class classification
    N = 10

    # Valid methods are: "RandomOver", "RandomUnder", "SMOTE" and "None"
    sampling_method = "None"
    print("sampling_method: %s" % sampling_method)
    print("Number of instances: %d" % len(Y))

    Matrix, NetworkTypeLabels, sum_accuracy, list_important_features = \
        sum_confusion_matrix(X, Y, sub_to_main_type, feature_order, isSubType, sampling_method, N)

    average_matrix = np.asarray(list(map(lambda row: list(map(lambda e: e / N, row)), Matrix)))
    print("average accuracy: %f" % (float(sum_accuracy) / float(N)))
    plot_feature_importance(list_important_features, feature_order)

    if not isSubType:
        sub_to_main_type = {v: v for v in sub_to_main_type.values()}
    plot_confusion_matrix(average_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)

    dist_matrix = make_symmetric(average_matrix)

    MDS_plot(dist_matrix, NetworkTypeLabels, sub_to_main_type)

    # construct an adjacency matrix from the aggrregated confusion matrix.
    adj_matrix = make_adj_matrix(average_matrix)

    G = nx.from_numpy_matrix(np.asarray(adj_matrix))
    graph_draw(G, NetworkTypeLabels, sub_to_main_type)

    # uncomment if want to save an unweighted network.
    # nx.write_edgelist(G, "G_%s.txt"%sampling_method)


if __name__ == '__main__':
    main()
