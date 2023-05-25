# Model design
import agentpy as ap
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


class Speaker(ap.Agent):

    def setup(self):
        """Initialize an agent with a variant A or B"""
        self.variant = np.random.choice([0, 1])   # variant1 = 0, variant2 = 1

    def communicate(self):
        """ Decide on a linguistic form based on the communication"""
        rn = np.random.rand()

        for n in self.network.neighbors(self):
            if n.variant == 0 and rn <= self.p.diffusion_rate:
                n.variant = self.variant
            else:
                n.variant = n.variant


class LangChangeModel(ap.Model):

    def setup(self):
        """ Initialize the agents of the model"""

        # Prepare a small-world network
        graph = nx.watts_strogatz_graph(
            self.p.num_agents,
            self.p.number_of_neighbors,
            self.p.network_randomness)

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.num_agents, Speaker)
        self.network = self.agents.network = ap.Network(self, graph)
        self.density = nx.density(graph)
        self.network.add_agents(self.agents, self.network.nodes)

    def update(self):
        """ Record variable after setup and each step. """
        n_agents = len(self.agents.select(self.agents.variant == 0))
        self['0'] = n_agents / self.p.num_agents
        self.record('0')
        self.record('1', 1-self['0'])
        self.record('Population', self.p.num_agents)
        self.record('Density', self.density)
        # self.record('Proportion', self.proportion)

        # color_dict = {0: 'b', 1: 'r'}
        # colors = [color_dict[c] for c in self.agents.variant]
        # nx.draw_circular(self.network.graph, node_color=colors, with_labels=False, font_weight='bold')
        # plt.show()

    def step(self):

        self.agents.communicate()


def langchange_stackplot(data, ax):
    """ Stackplot of people's condition over time. """

    x = data.index.get_level_values('t')
    y = [data[var] for var in ['0', '1']]
    sns.set()

    ax.stackplot(x, y, labels=['Variant1', 'Variant2'], colors=['r', 'b'])
    ax.legend()
    ax.set_xlim(0, max(1, len(x)-1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Percentage of population")


parameters = {'num_agents': 100,
              'diffusion_rate': 0.5,
              'steps': 50,
              'number_of_neighbors': 2,
              'network_randomness': 0.5
              }

model = LangChangeModel(parameters)
results = model.run()
# print(results)
# print(results.variables.LangChangeModel.keys())

fig, ax = plt.subplots()
langchange_stackplot(results.variables.LangChangeModel, ax)
plt.show()

# print(results.variables.Speaker[90:101])
# ax.set_title(f'The proportion of 0: {mdl.proportion}')
# color_dict = {0: 'b', 1: 'r'}
# colors = [color_dict[c] for c in mdl.agents.form]
# nx.draw_circular(mdl.network.graph, node_color=colors, with_labels=False, font_weight='bold')
