import igraph
import sys


def main(filepath):
    G = igraph.Graph.Read_Ncol(filepath, directed=False)
    vertices = G.community_fastgreedy(weights="weight").as_clustering()
    tuple_L = sum([map(lambda x: (x, i), L) for i, L in enumerate(list(vertices))], [])
    tuple_L_name = [(int(G.vs(id)["name"][0]), c) for id, c in tuple_L]
    tuple_L_sorted = sorted(tuple_L_name, key=lambda x: x[0])
    communities = [e[1] for e in tuple_L_sorted]
    print(communities)


if __name__ == '__main__':
    params = sys.argv
    filepath = params[1]
    main(filepath)
