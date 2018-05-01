import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from plot import index_to_color

colors_domain = ["#ff0000", "#9c8110", "#00d404", "#00a4d4", "#1d00d4", "#a400c3", "#831e1e"]

NetworkTypeLabels = ['Gene regulation', 'Proteins', 'Food Web', 'Fungal', 'Metabolic', 'Connectome', \
                     'PeerToPeer', 'Bayesian', 'Web Graph', 'Offline Social', 'Online Social', 'Affiliation', \
                     'Forest Fire Network', 'Scale Free', 'Small World', 'ER Network', 'Water Distribution',
                     'Software Dependency', \
                     'Communication', 'Digital Circuit', 'Subway', 'Roads']

NetworkTypeLabelsDomain = ["Biological", "Informational", "Social", "Synthesized", "Technological", "Transportation"]

sub_to_main_type = {'Relationships': 'Social', 'PeerToPeer': 'Informational', 'Forest Fire Network': 'Synthesized', \
                    'Scale Free': 'Synthesized', 'Offline Social': 'Social', 'Legal': 'Informational',
                    'Fiction': 'Social', \
                    'PowerGrid': 'Technological', 'Small World': 'Synthesized', 'Gene regulation': 'Biological',
                    'DistributWater ion': 'Technological', \
                    'Collaboration': 'Social', 'Bayesian': 'Informational', 'email': 'Social',
                    'Commerce': 'Informational', 'Water Distribution': 'Technological', \
                    'Language': 'Informational', 'Proteins': 'Biological', 'Software Dependency': 'Technological',
                    'Airport': 'Transportation', \
                    'Subway': 'Transportation', 'Food Web': 'Biological', 'Recommendations': 'Informational',
                    'Relatedness': 'Informational', \
                    'Online Social': 'Social', 'Fungal': 'Biological', 'Metabolic': 'Biological',
                    'Communication': 'Technological', \
                    'Web Graph': 'Informational', 'ER Network': 'Synthesized', 'Digital Circuit': 'Technological',
                    'Affiliation': 'Social', \
                    'Roads': 'Transportation', 'Connectome': 'Biological'}


# NetworkTypeLabels = ['Biological', 'Economic', 'Informational', 'Social', 'Synthesized', 'Technological', 'Transportation']

def read_community(file):
    result = []
    with open(file, "r") as f:
        for e in f.readlines():
            result.append(int(e))
    return result


def min_or_max(G, func=max):
    return func([attr["weight"] for i, j, attr in G.edges_iter(data=True)])


def threshold(G, alpha):
    thresholded_graph = nx.Graph()

    for u, v, w, in G.edges_iter(data='weight'):
        if w > alpha:
            thresholded_graph.add_edge(u, v, weight=w["weight"] * 50)

    return thresholded_graph


def graph_draw_helper(G, colors, pos, labels):
    minimum = min_or_max(G, min)  # the minimum of weights
    maximum = min_or_max(G, max)  # the maximum of weights
    n = maximum - minimum

    nx.draw_networkx_labels(G, pos=pos, alpha=0.5, labels=labels, font_size=14, font_family="sans-serif",
                            font_weight="medium")
    edge_alpha = map(lambda x: round(x, 4), np.linspace(0.25, 0.8, n))

    for e, v, w in list(G.edges_iter(data='weight')):
        # print "e,v,w:",(e,v,w)
        nx.draw_networkx_edges(G, pos=pos, edgelist=[(e, v)], alpha=0.55, width=w["weight"] * 0.225)

    nx.draw_networkx_nodes(G, pos=pos, nodelist=G.nodes(), alpha=0.55, linewidths=2.5, node_size=700, node_color=colors)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def graph_draw_community(G, pos, community):
    labels = {}
    for e in G.nodes():
        labels[e] = NetworkTypeLabels[e]

    # color_map = index_to_color(list(set(community)),"hsv")
    color_map = index_to_color([0, 1, 2, 3], "hsv")
    community = map(lambda c: c if not (c == 2 or c == 1) else 2 if c == 1 else 1, community)
    colors = [color_map(community[n]) for n in G.nodes()]

    graph_draw_helper(G, colors, pos, labels)


def graph_draw_domain(G, pos):
    labels = {}
    for e in G.nodes():
        labels[e] = NetworkTypeLabels[e]
        # labels[e] = NetworkTypeLabelsDomain[e]
    Domains = sorted(list(set(sub_to_main_type.values())))
    # color_map = index_to_color(Domains,"hsv")
    color_map = lambda i: colors_domain[i]

    colors = [color_map(Domains.index(sub_to_main_type[sub_domain])) for sub_domain in NetworkTypeLabels]

    graph_draw_helper(G, colors, pos, labels)
    # graph_draw_helper(G, colors_domain, pos, labels)


if __name__ == '__main__':
    # community = read_community("community.txt")
    G = nx.read_edgelist("G_sub_RandomUnder.txt", nodetype=int, data=(('weight', float),))
    G = threshold(G, 0.00)
    # pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.spring_layout(G)

    # position for domain 
    # pos = {0: np.array([ 0.46120474,  0.29051819]), 1: np.array([ 0.56478633,  0.1684904 ]), 2: np.array([ 1.        ,  0.11698007]), 3: np.array([ 0.4360418,  0.       ]), 4: np.array([ 0.31135404,  0.21310864]), 5: np.array([ 0.        ,  0.24437247])}

    # position for sub-domain 
    pos = {0: np.array([0.64359751, 0.38682058]), 1: np.array([0.42135993, 0.21089149]),
           2: np.array([0.6084664, 0.29737411]), 3: np.array([0.31535984, 0.46560666]),
           4: np.array([0.72710701, 0.29676786]), 5: np.array([0.35303137, 0.24453801]), 6: np.array([0., 0.35756103]),
           7: np.array([0.42715124, 0.37140814]), 8: np.array([0.58315806, 0.260424]),
           9: np.array([0.35592825, 0.16796377]), 10: np.array([0.23648181, 0.11115261]),
           11: np.array([0.29468445, 0.]), 12: np.array([0.25614322, 0.21568847]),
           13: np.array([0.69572081, 0.4206636]), 14: np.array([0.38646554, 0.06618521]),
           15: np.array([0.23926157, 0.38088921]), 16: np.array([0.32758902, 0.39871459]),
           17: np.array([0.47281187, 0.3235735]), 18: np.array([1., 0.39957405]),
           19: np.array([0.47323346, 0.41975049]), 20: np.array([0.41922152, 0.46998033]),
           21: np.array([0.25545288, 0.56398872])}

    # community = [0, 0, 0, 1, 0, 2, 3, 1, 0, 2, 2, 2, 2, 0, 0, 3, 1, 0, 0, 1, 1, 1] #None
    # community = [0, 0, 0, 1, 0, 2, 3, 1, 0, 2, 2, 2, 2, 0, 0, 3, 1, 0, 0, 1, 1, 1] # RandomOver
    community = [0, 1, 0, 2, 0, 1, 2, 2, 0, 1, 1, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 2]  # RandomUnder
    # community = [0, 0, 0, 1, 0, 2, 3, 1, 0, 2, 0, 2, 0, 1, 0, 3, 1, 0, 0, 1, 1, 1] # SMOTE
    graph_draw_community(G, pos, community)
    graph_draw_domain(G, pos)
