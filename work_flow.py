from __future__ import division
import math
import numpy as np
from preprocess import init
from multiclass import multiclass_classification
from plot import plot_confusion_matrix
from plot import plot_distance_matrix
from plot import matrix_clustering
from plot import plot_feature_importance
from plot import index_to_color
from plot import  MDS_plot

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx

import scipy.spatial.distance as ssd

import pylab
import scipy.cluster.hierarchy as sch

import networkx as nx


colors_domain = ["#ff0000", "#9c8110", "#00d404", "#00a4d4", "#1d00d4", "#a400c3", "#831e1e"]

def sum_confusion_matrix(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod, N):
    accum_matrix, NetworkTypeLabels, accum_acc, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod)
    list_important_features = [feature_importances]
    for i in range(N - 1):
        print "i: ",i
        cm, _, accuracy, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod)
        accum_matrix += cm
        accum_acc += accuracy
        list_important_features.append(feature_importances)
    return accum_matrix, NetworkTypeLabels, accum_acc, list_important_features


def make_symmetric(cm):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_filtered = map(lambda ax: map(lambda val: 0.0 if math.isnan(val) else val, ax),cm_normalized)
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
            cm_normalized_filtered[i][j] = (1 - cm_normalized_filtered[i][j])*100

    return np.asarray(cm_normalized_filtered)


def build_dendrogram(D, leave_name, sub_to_main_type, isSubType):
   
    Domains = list(set(sub_to_main_type.values()))
    color_map = index_to_color(Domains,"jet")
    fig = pylab.figure(figsize=(10, 10))
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
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
    fig.show()
    fig.savefig('only_dendrogram_.png', bbox_inches='tight')

def min_or_max(G,func=max):
    return func([attr["weight"] for i,j,attr in G.edges_iter(data=True)])


def threshold(G, alpha):
    thresholded_graph = nx.Graph()

    for u,v,w, in G.edges_iter(data='weight'):
        if w > alpha:
            thresholded_graph.add_edge(u,v, weight = w["weight"]*50)

    return thresholded_graph

def graph_draw(G, NetworkTypeLabels, sub_to_main_type):

    G = threshold(G,0.00)
    pos = nx.fruchterman_reingold_layout(G)
    #pos = nx.spring_layout(G)
    labels = {}
    for e in G.nodes():
        labels[e] = NetworkTypeLabels[e]

    Domains = list(set(sub_to_main_type.values()))

    #color_map = index_to_color(Domains,"hsv")
    color_map = lambda i: colors_domain[i]
    print NetworkTypeLabels
    print sub_to_main_type
    colors = [color_map(Domains.index(sub_to_main_type[sub_domain])) for sub_domain in NetworkTypeLabels]

    minimum = min_or_max(G,min) # the minimum of weights
    maximum = min_or_max(G,max)# the maximum of weights
    n = maximum - minimum

    nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=11)
    edge_alpha = map(lambda x:round(x,4), np.linspace(0.25, 0.8,n))
    
    for e,v,w in list(G.edges_iter(data='weight')):
        print "e,v,w:",(e,v,w)
        nx.draw_networkx_edges(G,pos=pos,edgelist=[(e,v)],alpha=0.6,width=w["weight"]*0.2)


    nx.draw_networkx_nodes(G,pos=pos,nodelist=G.nodes(),node_size=250,node_color=colors, alpha=0.6)

    plt.axis('off')
    plt.show()



def make_adj_matrix(cm):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_filtered = map(lambda ax: map(lambda val: 0.0 if math.isnan(val) else val, ax),cm_normalized)
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


def main():
    column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity","DegreeAssortativity","MeanGeodesicDistance","Diameter","MGD/Diameter",
                    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]

    column_names = ["NetworkType","SubType","ClusteringCoefficient","DegreeAssortativity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
   
    
    isSubType = False
    at_least = 1#6
    X,Y,sub_to_main_type, feature_order= init("features.csv", column_names, isSubType, at_least)
    N = 1000
    sampling_method = "RandomUnder"
    print "sampling_method: %s"%sampling_method
    print "Number of instances: %d"%len(Y)

    
    Matrix, NetworkTypeLabels, sum_accuracy, list_important_features = sum_confusion_matrix(X, Y, sub_to_main_type, feature_order, isSubType, sampling_method, N)
    average_matrix = np.asarray(map(lambda row: map(lambda e: e/N ,row), Matrix))
    print "average accuracy: %f"%(float(sum_accuracy)/float(N))
    #plot_feature_importance(list_important_features, feature_order)
    if not isSubType:
        sub_to_main_type = {v:v for v in sub_to_main_type.values()}
    plot_confusion_matrix(average_matrix, NetworkTypeLabels, sub_to_main_type, isSubType,"confusion_%s.png"%sampling_method)
    dist_matrix = make_symmetric(average_matrix)
    

    #MDS_plot(dist_matrix, NetworkTypeLabels, sub_to_main_type)

    adj_matrix = make_adj_matrix(average_matrix)
    
    G = nx.from_numpy_matrix(np.asarray(adj_matrix))
    #sub_to_main_type = {v:v for v in sub_to_main_type.values()}
    #graph_draw(G, NetworkTypeLabels, sub_to_main_type)
    #print NetworkTypeLabels
    nx.write_edgelist(G, "G_%s.txt"%sampling_method)
    
    #plot_distance_matrix(dist_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)
    #build_dendrogram(dist_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)
        
if __name__ == '__main__':
    main()