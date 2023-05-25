# Model design
import copy
import agentpy as ap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def animation_plot(m):
    plt.title(f"v1 distribution: {round(m.average_belief, 2)}")
    color_dict = {'v1': 'b', 'v2': 'r', '': 'g'}
    colors = [color_dict[c] for c in m.agents.sampled_token]
    nx.draw_circular(m.network.graph, node_color=colors)


def batch_simulate(num_sim, params):

    for i in range(0, num_sim):
        model_i = LangChangeModel(parameters)
        results_i = model_i.run()
        data = results_i.variables.LangChangeModel
        data['average_belief'].plot(linewidth=0.8)
        # data['average_sample_store'].plot(label=f'sim{i}', linewidth=0.8)
        # data['average_sigmoid_belief'].plot(label=f'sim{i}', linewidth=0.8)

    plt.ylim((0, 1))
    plt.xlabel('t')
    plt.text(params["steps"]-50, 0.75, f'agents = {params["agents"]}\n'
                         f'neighbors = {params["number_of_neighbors"]}\n'
                         f'network = {params["network_density"]}\n'
                         f'steps = {params["steps"]}\n'
                         f'simulations = {num_sim}',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.5),
             fontdict={'family': 'serif', 'size': 9, 'color': 'black'})
    # plt.legend(loc='upper right')
    plt.show()


class Speaker(ap.Agent):

    def setup(self):
        """Initialize agent's parameters"""

        # initial distribution of v1 and v2 based on prior belief
        # self.r_choice = np.random.choice([1, 0], p=[0.8, 0.2])
        # self.store = np.random.choice(['v1', 'v2'], size=10, p=[self.r_choice, 1-self.r_choice])
        # self.store = np.random.choice(['v1', 'v2'], size=10, p=[0.5, 0.5])
        self.store = np.random.choice(['v1', 'v2'], size=10)
        # self.belief = 0.9

        # prior belief of choosing v1
        self.belief = np.count_nonzero(self.store == 'v1') / len(self.store)
        # self.store = np.random.choice(['v1', 'v2'], size=10, p=[self.belief, 1-self.belief])

        # the copy of the initial store
        # will be updated during interactions
        self.updated_store = copy.deepcopy(self.store)

        # the copy of the initial belief
        # will be updated during interactions
        self.updated_belief = self.belief

        # Define the preference variable (preference for v1)
        # corresponds with a probability of using v1
        # self.preference = self.updated_belief
        # self.v1_probability = sigmoid(self.preference)

        self.sample_store = np.concatenate((self.store, self.updated_store))

        # sampled token
        self.sampled_token = ''


    def interact(self):
        """ Define the communication behaviour of speaker and other speaker
        in the network"""

        for j in self.network.neighbors(self):
            # A variant is sampled from each store

            # sample_store = np.concatenate((self.updated_store, self.store))
            # token_speaker = np.random.choice(sample_store, size=1)
            token_speaker = np.random.choice(self.updated_store, size=1)
            self.sampled_token = token_speaker[0]
            # j_sample_store = np.concatenate((j.updated_store, j.store))
            # token_other_speaker = np.random.choice(j_sample_store, size=1)
            token_other_speaker = np.random.choice(j.updated_store, size=1)
            j.sampled_token = token_other_speaker[0]

            # Listening to herself or reinforcing her own behaviour
            # Copying a sampled variant and replacing it within the same store that the sample was taken from
            reinforce_speaker = np.random.choice([i for i in range(0, len(self.updated_store))], size=1)
            self.updated_store[reinforce_speaker] = token_speaker

            # if token_speaker[0] == 'v1':
            #     self.preference += self.p.doubt_step
            # else:
            #     self.preference -= self.p.doubt_step

            reinforce_other_speaker = np.random.choice([i for i in range(0, len(j.updated_store))], size=1)
            j.updated_store[reinforce_other_speaker] = token_other_speaker

            # if token_other_speaker[0] == 'v1':
            #     j.preference += self.p.doubt_step
            # else:
            #     j.preference -= self.p.doubt_step

            # Modifying the behaviour to match more closely that of the interlocutor
            # Placing a copy of the other speaker's token in his store
            modify_speaker = np.random.choice([i for i in range(0, len(self.updated_store))], size=1)
            self.updated_store[modify_speaker] = token_other_speaker

            # if token_other_speaker[0] == 'v1':
            #     self.preference += self.p.doubt_step
            # else:
            #     self.preference -= self.p.doubt_step

            modify_other_speaker = np.random.choice([i for i in range(0, len(j.updated_store))], size=1)
            j.updated_store[modify_other_speaker] = token_speaker

            # if token_speaker[0] == 'v1':
            #     j.preference += self.p.doubt_step
            # else:
            #     j.preference -= self.p.doubt_step

    def update(self):
        """ Update and record belief of choosing v1"""
        self.updated_belief = np.count_nonzero(self.updated_store == 'v1') / len(self.updated_store)
        # self.v1_probability = sigmoid(self.preference)
        # self.sample_store = np.concatenate((self.store, self.updated_store))
        self.record('belief', self.updated_belief)
        self.record('initial_belief', self.belief)
        # self.record('sigmoid_belief', self.v1_probability)
        # self.record('preference', self.preference)
        # self.record('sample_store', self.sample_store)


class LangChangeModel(ap.Model):

    def setup(self):

        # Initialize a population of agents
        # Prepare a small-world network
        graph = nx.watts_strogatz_graph(
            self.p.agents,
            self.p.number_of_neighbors,
            self.p.network_density
        )

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Speaker)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

        # self.plot_graph = nx.draw_circular(self.network.graph)

    def update(self):
        """ Record variables after setup and each step"""

        # Record average belief after each simulation step
        self.average_belief = sum(self.agents.updated_belief) / len(self.agents.updated_belief)
        self.record('average_belief', self.average_belief)

        # self.average_sigmoid_belief = sum(self.agents.v1_probability) /len(self.agents.v1_probability)
        # self.record('average_sigmoid_belief', self.average_sigmoid_belief)

        # self.average_preference = sum(self.agents.preference) / len(self.agents.preference)
        # self.record('average_preference', self.average_preference)

        # self.average_sample_store = np.count_nonzero(self.agents.sample_store == 'v1') / np.count_nonzero(self.agents.sample_store)
        # self.record('average_sample_store', self.average_sample_store)

    def step(self):

        # Choose two agents to interact
        # i, j = self.agents.random(2)

        # Make sure the same agent is not selected twice
        # while i == j:
        #     j = self.agents.random(1)

        # Call the interaction function
        # i.interact(j)
        self.agents.interact()

        # Call the update function
        # i.update()
        # j.update()
        self.agents.update()

    # def end(self):

        # plt.show()


# Set up parameters for the model
parameters = {'agents': 100,
              'number_of_neighbors': 8,
              'network_density': 1,
              'doubt_step': 0.01,
              'steps': 1500,
              }

# random.seed(123)
# Initialize and run the model
model = LangChangeModel(parameters)
results = model.run()
# print(results.variables.Speaker)

# print(results.variables.LangChangeModel)
# exit()
# print(results.variables.Speaker)
# results.variables.LangChangeModel.plot(x='average_preference', y='average_sigmoid_belief')
# plt.show()
# exit()

# Store results in a variable
average_belief = results.variables.LangChangeModel
# average_belief.to_excel('data_output.xlsx')
# exit()

# fig, axs = plt.subplots()
# anim = ap.animate(LangChangeModel(parameters), fig, axs, animation_plot)

# Demonstrate results
# print(average_belief.head())
# print(average_belief.tail())

# Plot average belief over simulation steps
# average_belief.plot()
# plt.ylim((0, 1))
# plt.show()

# Run N simulations
batch_simulate(num_sim=20, params=parameters)