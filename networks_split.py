import matplotlib.pyplot as plt
import networkx as nx


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


print(split_population(graph=nx.watts_strogatz_graph(10, 2, 0.5)))