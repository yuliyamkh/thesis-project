
import networkx as nx
import community
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cm as cm
import numpy as np
from scipy.cluster import hierarchy


def split_network(graph):

    if len(graph.nodes) > 10:
        coms = nx.community.louvain_communities(graph, seed=123)

        for com in coms:
            subpop = graph.subgraph(com)
            nx.draw(subpop, with_labels=True)
            plt.show()
            split_network(subpop)


def split_population(graph):

    subpops = []

    if len(graph.nodes) >= 5:
        # Split graph into groups
        coms = nx.community.louvain_communities(graph, seed=123)

        for com in coms:
            subpop = graph.subgraph(com)
            subpops.append(subpop)

    return subpops


G = nx.watts_strogatz_graph(10, 5, 0.5)
dendo = nx.community.louvain_communities(G)
print(list(dendo))

