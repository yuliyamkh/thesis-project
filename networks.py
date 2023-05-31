import networkx as nx
import matplotlib.pyplot as plt

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