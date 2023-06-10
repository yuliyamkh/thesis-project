import networkx as nx
import matplotlib.pyplot as plt


Graph = nx.watts_strogatz_graph(100, 2, 0.5)


def connect_new_nodes(graph, new_nodes):
    # Get the existing nodes before the new nodes were added
    existing_nodes = list(set(graph.nodes) - set(new_nodes))

    # Iterate through the new nodes
    for new_node in new_nodes:
        # Connect the new node to all existing nodes
        for existing_node in existing_nodes:
            graph.add_edge(new_node, existing_node)


def add_new_nodes(graph, num_nodes):
    # Find the current highest node number in the graph
    max_node = max(graph.nodes)

    # Add new nodes with increasing numbers
    new_nodes = range(max_node + 1, max_node + 1 + num_nodes)
    connect_new_nodes(graph, new_nodes)
    # graph.add_nodes_from(new_nodes)


def recursive_partition(G, min_nodes=10, parent=None, partition_tree=None):
    if partition_tree is None:
        partition_tree = {}

    if len(G.nodes) > min_nodes:
        partition = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(G)
        subgraphs = [G.subgraph(nodes) for nodes in partition]
        partition_tree[parent] = subgraphs

        for subgraph in subgraphs:
            recursive_partition(subgraph, min_nodes, parent=subgraph, partition_tree=partition_tree)

        return partition_tree


tr = recursive_partition(Graph)


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.data) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


root = TreeNode(list(Graph.nodes()))
graph_1 = TreeNode(list(tr[None][0].nodes()))
graph_2 = TreeNode(list(tr[None][1].nodes()))

root.add_child(graph_1)
root.add_child(graph_2)

graph_1_1 = TreeNode(list(tr[tr[None][0]][0].nodes()))
graph_1_2 = TreeNode(list(tr[tr[None][0]][1].nodes()))

graph_1.add_child(graph_1_1)
graph_1.add_child(graph_1_2)

graph_2_1 = TreeNode(list(tr[tr[None][1]][0].nodes()))
graph_2_2 = TreeNode(list(tr[tr[None][0]][1].nodes()))

graph_2.add_child(graph_2_1)
graph_2.add_child(graph_2_2)

print(root)
exit()

from graphviz import Digraph

def visualize_tree(node, graph=None):
    if graph is None:
        graph = Digraph('G', filename='tree.gv')
        graph.node(str(id(node)), str(node.data))

    for child in node.children:
        graph.node(str(id(child)), str(child.data))
        graph.edge(str(id(node)), str(id(child)))
        visualize_tree(child, graph)

    return graph

print(visualize_tree(root).view())
exit()

n = 10    # Number of nodes
k = 6  # Each node is connected to k nearest neighbors
rewiring_probs = [0, 0.1, 0.3, 0.5, 0.7, 1]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
ax = axes.flatten()

# Generate and draw the networks
for i, p in enumerate(rewiring_probs):
    G = nx.watts_strogatz_graph(n, k, p)
    # nx.draw(G, node_color='blue', edge_color='gray', node_size=8)
    nx.draw_circular(G, ax=ax[i], node_size=8, with_labels=False)
    average_path_length = round(nx.average_shortest_path_length(G), 2)
    average_clustering_coefficient = round(nx.average_clustering(G), 2)
    ax[i].set_axis_off()
    ax[i].set_title(f'Rewiring Probability: {p}\n'
                    f'Average Path Length: {average_path_length}\n'
                    f'Average Clustering Coefficient: {average_clustering_coefficient}',
                    fontsize=7)

plt.show()

avg_shortest_path_lengths = []
clustering_coefficients = []

for p in rewiring_probs:
    G = nx.watts_strogatz_graph(n, k, p)
    avg_shortest_path_length = nx.average_shortest_path_length(G)
    clustering_coefficient = nx.average_clustering(G)

    avg_shortest_path_lengths.append(avg_shortest_path_length)
    clustering_coefficients.append(clustering_coefficient)

plt.plot(rewiring_probs, avg_shortest_path_lengths, label='Average Shortest Path Length')
plt.plot(rewiring_probs, clustering_coefficients, label='Clustering Coefficient')
plt.xlabel('Rewiring Probability')
plt.ylabel('Network Metrics')
plt.legend()
plt.show()

""" Note:
As the rewiring probability increases, the average shortest path length generally decreases, 
indicating that the network becomes more interconnected and information can spread more quickly. 

On the other hand, the clustering coefficient also decreases, suggesting that the local connectivity
of the network weakens as the rewiring probability increases. 

This trade-off between local and long-range connectivity can be analyzed to understand the impact 
of the rewiring probability on the network's properties and behavior.
"""