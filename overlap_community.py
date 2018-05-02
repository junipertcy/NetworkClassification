import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

NetworkTypeLabels = ['Gene regulation', 'Proteins', 'Food Web', 'Fungal', 'Metabolic', 'Connectome',
                     'PeerToPeer', 'Bayesian', 'Web Graph', 'Offline Social', 'Online Social', 'Affiliation',
                     'Forest Fire Network', 'Scale Free', 'Small World', 'ER Network', 'Water Distribution',
                     'Software Dependency',
                     'Communication', 'Digital Circuit', 'Subway', 'Roads']

N = len(NetworkTypeLabels)

none = [0, 0, 0, 1, 0, 2, 3, 1, 0, 2, 2, 2, 2, 0, 0, 3, 1, 0, 0, 1, 1, 1]
random_under = [0, 1, 0, 2, 0, 1, 2, 2, 0, 1, 1, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 2]
random_over = [0, 0, 0, 1, 0, 2, 3, 1, 0, 2, 2, 2, 2, 0, 0, 3, 1, 0, 0, 1, 1, 1]
smote = [0, 0, 0, 1, 0, 2, 3, 1, 0, 2, 0, 2, 0, 1, 0, 3, 1, 0, 0, 1, 1, 1]


def matrix_clustering(D, leave_name):
    f, ax = plt.subplots()
    # Compute first dendrogram.
    Y = sch.linkage(D)
    Z1 = sch.dendrogram(Y, no_plot=True)

    # Compute second dendrogram.
    Y = sch.linkage(D)
    Z2 = sch.dendrogram(Y, no_plot=True)

    # Plot distance matrix.

    idx1 = Z1['leaves']
    idx2 = Z2['leaves']

    D = D[idx1, :]
    D = D[:, idx2]

    for i in range(N):
        for j in range(N):
            D[i][j] = abs(D[i][j] - 4)

        cmap = plt.get_cmap('Blues', 5)  # plt.cm.Blues
    im = ax.imshow(D, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0 - .5, vmax=4 + .5)

    # mapping from an index to an axis label (gml file name, NetworkType, SubType)
    axis_labels = [leave_name[i] for i in idx1]

    tick_marks = np.arange(len(axis_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(axis_labels, rotation=90)
    ax.set_yticklabels(axis_labels)

    # Plot colorbar.
    f.colorbar(im, ticks=np.arange(0, 5))
    f.tight_layout()

    plt.show()
    plt.savefig('overlaps.png', bbox_inches='tight')


def count_overlaps(L1, L2):
    return sum([1 if l1 == l2 else 0 for l1, l2 in zip(L1, L2)])


def main():
    matrix = np.zeros((N, N))
    combined = zip(none, random_over, random_under, smote)

    for first, second in itertools.permutations(NetworkTypeLabels, 2):
        combined = list(zip(none, random_over, random_under, smote))
        i = NetworkTypeLabels.index(first)
        j = NetworkTypeLabels.index(second)
        e_i = combined[i]
        e_j = combined[j]

        print(first, second)
        print(e_i, e_j)

        num_overlaps = count_overlaps(e_i, e_j)
        print(num_overlaps)
        print("-------------------")
        matrix[i][j] = 4 - num_overlaps

    # for i in range(N):
    # 	matrix[i][i] = 4

    matrix_clustering(matrix, NetworkTypeLabels)


if __name__ == '__main__':
    main()
