import math
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from preprocess import *

# colors each corresponds to network domain
colors_domain = ["#ff0000", "#9c8110", "#00d404", "#00a4d4", "#1d00d4", "#a400c3", "#831e1e"]


def isFloat(x):
    """checks if the input x is float number"""
    try:
        float(x)
        return True
    except ValueError:
        return False


def index_to_color(iterator, colotType):
    """
    :param iterator: some iterator (e.g. list of strings).
    :param colotType: a string indicating color type (e.g. "jet").
    :return:
        a function that maps an index in the given iterator to a color in the color map.
    """
    jet = plt.get_cmap(colotType)
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(iterator))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return lambda i: scalarMap.to_rgba(i)

def sort_by_feature(network_dict, feature_names):
    result = []
    for k in network_dict.keys():
        each = []
        for name in feature_names:
            entry = network_dict[k][name]
            if isFloat(entry):
                each.append(float(entry))
            else:
                each.append(entry)
        result.append(tuple(each))
    return result

def normalize_mgd(network_tuple):
    return map(lambda (x1,x2,x3,x4,x5): (x1,x2,x3,x4/math.log(x5)) ,network_tuple)

def plot_3d(data, feature_names):
    ts = [t for t,f1,f2,f3 in data]

    colors = ["y","r","c","b","g","k","m"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t,c in zip(set(ts),colors):
        xs = [f1 for network_type,f1,f2,f3 in data if network_type == t]
        ys = [f2 for network_type,f1,f2,f3 in data if network_type == t]
        zs = [f3 for network_type,f1,f2,f3 in data if network_type == t]
        
        ax.plot(xs, ys, zs,"o",c=c,label=t,alpha=0.85)
    
    ax.set_xlabel(feature_names[1])
    ax.set_ylabel(feature_names[2])
    ax.set_zlabel(feature_names[3])
    #ax.set_zscale("log")
    ax.legend(loc = 'upper left')
    plt.draw()
    plt.show()


def make_mask(matrix):
    matrix = np.array(matrix)
    z = np.zeros(matrix.shape)
    for i, row in enumerate(matrix):
        for j, e_ij in enumerate(row):
            if e_ij == 0:
                z[i][j] = 1
    return z

def plot_confusion_matrix(cm, NetworkTypeLabels, sub_to_main_type, isSubType,filename=None):

    #norm = matplotlib.colors.BoundaryNorm(bounds,plt.cm.Blues.N)
    Domains = sorted(list(set(sub_to_main_type.values())))
    #color_map = index_to_color(Domains,"jet")
    color_map = lambda i: colors_domain[i]
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_filtered = map(lambda ax: map(lambda val: 0.0 if math.isnan(val) else val, ax),cm_normalized)

    f, ax = plt.subplots()#figsize=(8, 7.5))
    mask = make_mask(cm_normalized_filtered)
    masked_array = np.ma.array(cm_normalized_filtered, mask=mask)
    cmap = matplotlib.cm.jet
    cmap.set_bad('white',1.)
    im = ax.imshow(masked_array, interpolation='nearest', cmap=cmap)

    if isSubType:
        prev = sub_to_main_type[NetworkTypeLabels[0]]
        i = -0.5
        for t in NetworkTypeLabels:
            if prev != sub_to_main_type[t]:
                ax.axhline(i,color="k")
                ax.axvline(i,color="k")
                prev = sub_to_main_type[t]
            i += 1


    # dim = len(cm)
    # for i in range(dim):
    #     for j in range(dim):
    #         if cm[i][j] != 0.0:
    #             ax.text(j, i, cm[i][j], va='center', ha='center', color = "r", size=8)
    
    #ax.set_title("Confusion Matrix")
    f.colorbar(im)
    tick_marks = np.arange(len(NetworkTypeLabels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(NetworkTypeLabels, rotation=90)
    ax.set_yticklabels(NetworkTypeLabels)

    xlbls = ax.get_xmajorticklabels()
    ylbls = ax.get_ymajorticklabels()

    if isSubType or True:
        for ylbl in ylbls:
            domain = sub_to_main_type[ylbl.get_text()]
            index = Domains.index(domain)
            ylbl.set_color(color_map(index))

        for xlbl in xlbls:
            domain = sub_to_main_type[xlbl.get_text()]
            index = Domains.index(domain)
            xlbl.set_color(color_map(index))


    f.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')


def plot_distance_matrix(distance_m, NetworkTypeLabels, sub_to_main_type, isSubType):
    
    f, ax = plt.subplots()
    im = ax.imshow(distance_m, interpolation='nearest', cmap=plt.cm.Blues)

    if isSubType:
        prev = sub_to_main_type[NetworkTypeLabels[0]]
        i = -0.5
        for t in NetworkTypeLabels:
            if prev != sub_to_main_type[t]:
                ax.axhline(i)
                ax.axvline(i)
                prev = sub_to_main_type[t]
            i += 1

    # dim = len(distance_m)
    # for i in range(dim):
    #   for j in range(dim):
    #       #if distance_m[i][j] != 0.0:
    #       ax.text(j, i, round(distance_m[i][j],2), va='center', ha='center', color = "r")
    
    f.colorbar(im)
    tick_marks = np.arange(len(NetworkTypeLabels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(NetworkTypeLabels, rotation=90,fontsize=10)
    ax.set_yticklabels(NetworkTypeLabels, fontsize=10)
    #f.tight_layout()
    plt.show()


def MDS_plot(distance_matrix, NetworkTypeLabels, sub_to_main_type):
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(distance_matrix).embedding_
    clf = PCA(n_components=2)
    pos = clf.fit_transform(pos)

    xs = [x for x,y in pos]
    ys = [y for x,y in pos]
    plt.scatter(xs, ys)
    Domains = list(set(sub_to_main_type.values()))
    color_map = index_to_color(Domains,"hsv")

    for i,type_label in enumerate(NetworkTypeLabels):
        domain = sub_to_main_type[type_label]
        index = Domains.index(domain)
        plt.annotate(type_label,xy = (xs[i], ys[i]),color=color_map(index))

    plt.axhline(0)
    plt.axvline(0)
    plt.show()


def plot_2d(X, Y, x_index, y_index, x_label, y_label, xlog_scale=False,  ylog_scale=False):


    X = np.array(X)
    Y = np.array(Y)
    
    ts = set(Y)
    values = range(len(ts))
    
    color_map = index_to_color(ts,"hsv")

    color_select = lambda x: "r" if x == "Other" else "b"
    marker_select = lambda x: "o" if x == "Other" else ","

    ax = plt.subplot(111)
    
    for idx,label in enumerate(ts):

        if len(ts) > 2:
            colorVal = color_map(values[idx])
            plt.scatter(x=X[:,x_index][Y == label],
                        y=X[:,y_index][Y == label],
                        color=colorVal,
                        alpha=0.6,
                        label=label,
                        marker=marker_select(label),
                        s=60,
                        edgecolors="k"
                        )
        else:

            plt.scatter(x=X[:,x_index][Y == label],
                        y=X[:,y_index][Y == label],
                        color=color_select(label),
                        alpha=0.6,
                        label=label,
                        marker=marker_select(label),
                        s=60,
                        edgecolors="k"
                        )

    plt.xlabel(x_label,fontsize = 20)
    plt.ylabel(y_label,fontsize = 20)

    if xlog_scale:
        plt.xscale("log")

    if ylog_scale:
        plt.yscale("log")

    plt.legend(loc='upper right', fancybox=True,prop={'size':15})#,bbox_to_anchor=(1.1, 1.05))
    
    plt.tight_layout
    plt.show()


def plot_scikit_lda(X, Y):
    ts = set(Y)
    values = range(len(ts))
    
    color_map = index_to_color(ts, "hsv")


    ax = plt.subplot(111)
    for idx,label in enumerate(ts):
        colorVal = color_map(values[idx])

        plt.scatter(x=X[:,0][Y == label],
                    y=X[:,1][Y == label] * (-1), # flip the figure
                    color=colorVal,
                    alpha=0.8,
                    label=label
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.1, 1.05),prop={'size':10})
    
    plt.tight_layout
    plt.show()

def plot_scikit_lda_3d(X, Y):
    ts = set(Y)
    values = range(len(ts))
    jet = plt.get_cmap('jet') 
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(ts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx,t in enumerate(ts):
        colorVal = scalarMap.to_rgba(values[idx])
        ax.plot(X[:,0][Y == t], 
                X[:,1][Y == t], 
                X[:,2][Y == t],"o",c=colorVal,label=t,alpha=0.85)
    
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    #ax.set_zscale("log")
    ax.legend(loc = 'upper right',bbox_to_anchor=(1.1, 1.05),prop={'size':10})
    plt.draw()
    plt.show()



def matrix_clustering(D, leave_name):
    import pylab
    import scipy.cluster.hierarchy as sch
    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(10,10))
    ax1 = fig.add_axes([0.00, 0.1, 0.2, 0.6])
    Y = sch.linkage(D)#, method='centroid')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.05])
    Y = sch.linkage(D)#, method='centroid')
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    print D
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)

    # mapping from an index to an axis label (gml file name, NetworkType, SubType)
    axis_labels = [leave_name[i] for i in idx1]; print idx1

    tick_marks = np.arange(len(axis_labels))
    axmatrix.yaxis.set_label_position('right')
    axmatrix.set_yticks(tick_marks)
    axmatrix.set_yticklabels(axis_labels)
    pylab.yticks(fontsize=7)
    
    #Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    fig.show()
    fig.savefig('dendrogram.png',bbox_inches='tight')


def plot_feature_importance(Ls, feature_order):
    Ls = map(lambda x: map(lambda y: y[1],x), Ls)
    Ls = zip(*Ls) # Ls = [("ClustersingCoefficient", "ClustersingCoefficient", ..),]
    
    freq = {f: [] for f in feature_order}

    for freq_fs in Ls:
        for f in feature_order:
            freq[f].append(freq_fs.count(f))

    
    color_map = index_to_color(freq,"jet")

    iterate = sorted(list(freq.keys()),key=lambda x: x,reverse=True)

    first = iterate[0]
    colorVal = color_map(0)
    p = plt.bar(range(1,len(feature_order)+1), freq[first],  0.35, color=colorVal)
    prev = freq[first] # previous stack

    ps = [p] # storing axis objects
    who_is_dominant = [map(lambda x: (first,x),freq[first])]

    for i,k in enumerate(iterate[1:]):
        colorVal = color_map(i+1)
        p = plt.bar(range(1,len(feature_order)+1), freq[k],  0.35, color=colorVal, bottom=prev)
        who_is_dominant.append(map(lambda x: (k,x),freq[k]))
        prev = map(lambda x: x[0]+x[1] , zip(prev,freq[k]))
        ps.append(p)

    ranking = []

    for rank in zip(*who_is_dominant):
        print sorted(rank, key=lambda x:x[1],reverse=True)
        ranking.append(sorted(rank, key=lambda x:x[1],reverse=True))


    plt.legend(ps,iterate, bbox_to_anchor=(1.3, 0.4),prop={'size':16})
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xlim(0.75,8.5)
    plt.xlabel('Feature Importance',fontsize = 22)
    plt.ylabel('Frequency', fontsize = 22)

    plt.show()

    return ranking


def main():
    types_tobe_extracted = ["Biological","Technological"]
    feature_names = ["NetworkType","DegreeAssortativity","ClusteringCoefficient","MGD/Diameter"]
    network_dict = data_read("features.csv", *feature_names, types=types_tobe_extracted)

    network_tuple = sort_by_feature(network_dict, feature_names)

    plot_3d(network_tuple, feature_names)


def main2():
    x_label = "NumberOfNodes"
    y_label = "MeanDegree"
    column_names = ["NetworkType","SubType",x_label,y_label]
    exclusive_types = ["Economic", "Synthesized"]
    isSubType = False
    at_least = 1
    X,Y, sub_to_main_type, feature_order = init("features.csv", column_names, isSubType, at_least, exclusive_types=exclusive_types)

    x_index = feature_order.index(x_label)
    y_index = feature_order.index(y_label)


    plot_2d(X, Y, x_index, y_index, x_label, y_label, xlog_scale=True, ylog_scale=True)
   

if __name__ == '__main__':
    main2()


