# Model design
import copy
import agentpy as ap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def batch_simulate(num_sim, params):

    fig, ax = plt.subplots()

    for i in range(0, num_sim):
        print(f'Simulation number: {i}')
        model_i = LangChangeModel(parameters)
        results_i = model_i.run()
        data = results_i.variables.LangChangeModel
        data['average_belief'].plot(linewidth=0.8)

    plt.ylim((0, 1))
    plt.xlabel('t')

    text = (f'agents = {params["agents"]}\n'
                         f'neighbors = {params["number_of_neighbors"]}\n'
                         f'network = {params["network_density"]}\n'
                         f'steps = {params["steps"]}\n'
                         f'simulations = {num_sim}')

    ax.annotate(text, xy=(1, 1), xytext=(-100, -15), fontsize=8,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='left', verticalalignment='top')

    plt.show()


def selection(relative_freq_innovation, b):
    """
    Generate positive replicator selection
    :param relative_freq:
    :return:
    """
    updated_relative_freq_innovation = relative_freq_innovation

    if 0 <= relative_freq_innovation <= 1 / (1 + b):
        updated_relative_freq_innovation = (1 + b) * relative_freq_innovation
    if 1 / (1+b) <= relative_freq_innovation <= 1:
        updated_relative_freq_innovation = 1

    return updated_relative_freq_innovation


class Agent(ap.Agent):

    def setup(self) -> None:
        """
        Set up an agent's initial states.
        This method is called by the model during the setup phase,
        before the simulation starts
        """

        # Agent's grammar consisting of V = 2 linguistic variants
        lingueme = ['v1', 'v2']

        # The number indicating the size of the store
        memory_size = 10

        # Initial distribution of v1 and v2 based on prior belief
        self.store = np.random.choice(lingueme, size=memory_size)
        # self.store = np.random.choice(['v1', 'v2'], size=10, p=[0.01, 1-0.01])

        # Prior belief of choosing v1
        self.belief = np.count_nonzero(self.store == 'v1') / len(self.store)

        # The copy of the initial store will be updated during interactions
        self.updated_store = copy.deepcopy(self.store)

        # The copy of the initial belief will be updated during interactions
        self.updated_belief = copy.deepcopy(self.belief)

        # The produced utterance
        self.sampled_token = ''

    def speak(self) -> None:
        """
        Produce an utterance by sampling one token from the store
        """

        # probs = np.array([self.belief, 1-self.belief])
        # probs[probs < 0] = 0
        # probs = probs / np.sum(probs)
        # self.sampled_token = np.random.choice(self.updated_store)
        self.sampled_token = np.random.choice(['v1', 'v2'], size=1, p=[self.updated_belief, 1-self.updated_belief])[0]
        # self.sampled_token = np.random.choice(['v1', 'v2'], size=1, p=list(probs))[0]

    def reinforce(self) -> None:
        """
        Reinforce the own behaviour replacing a randomly
        removed token in the store with the sampled
        token's copy
        """

        # Choose a random index to remove
        random_index = np.random.randint(len(self.updated_store))

        # Remove the element at the random index
        self.updated_store = np.delete(self.updated_store, random_index)

        # Append the sampled token
        self.updated_store = np.append(self.updated_store, self.sampled_token)

        # Selection
        # if self.sampled_token == 'v1':
        #     self.belief = selection(self.belief, b=0.001)

    def listen(self, neighbour) -> None:
        """
        Listen to the neighbour to match more closely his behaviour
        Replacing the randomly removed token in the story with the
        neighbour's sampled token
        :param neighbour: one of the k agent's neighbours
        """

        # Choose a random index to remove
        randon_index = np.random.randint(len(self.updated_store))

        # Remove the element at the random index
        self.updated_store = np.delete(self.updated_store, randon_index)

        # Append the neighbour's sampled token
        self.updated_store = np.append(self.updated_store, neighbour.sampled_token)

        # Selection
        # if neighbour.sampled_token == 'v1':
        #     self.belief = selection(self.belief, b=0.001)

    def update(self):
        """
        Record belief of choosing the innovation variant v1
        based on the updated store.
        """

        self.updated_belief = selection(np.count_nonzero(self.updated_store == 'v1') / len(self.updated_store), b=0.001)
        self.record('belief', self.updated_belief)
        self.record('initial_belief', self.belief)


class LangChangeModel(ap.Model):

    def setup(self) -> None:
        """
        Initialize a population of agents and
        the network in which they exist and interact
        """

        graph = nx.watts_strogatz_graph(
            self.p.agents,
            self.p.number_of_neighbors,
            self.p.network_density
        )

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Agent)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

    def update(self):
        """
        Record variables after setup and each step
        """

        # Record average belief after each simulation step
        average_belief = sum(self.agents.updated_belief) / len(self.agents.updated_belief)
        self.record('average_belief', average_belief)

    def step(self):
        """
        Choose two agents who are in a neighborhood
        to each other to interact and perform the actions
        of speaking, reinforcing, and listening
        """

        # Choose a random agent from agents
        agent = self.random.choice(self.agents)

        # Initialize neighbors
        neighbors = [j for j in self.network.neighbors(agent)]

        # Select one random neighbor
        neighbor = self.random.choice(neighbors)

        agent.speak()
        neighbor.speak()

        agent.reinforce()
        neighbor.reinforce()

        agent.listen(neighbor)
        neighbor.listen(agent)

        agent.update()
        neighbor.update()

    def end(self):
        """
        Record evaluation measures at the end of the simulation.
        """
        final_average_belief = sum(self.agents.updated_belief) / len(self.agents.updated_belief)
        self.report('Final_average_belief', final_average_belief)


# Set up parameters for the model
parameters = {'agents': 50,
              'number_of_neighbors': 5,
              'network_density': 0.5,
              'steps': 200000
              }

model = LangChangeModel(parameters)
results = model.run()
# Run N number of simulations
batch_simulate(num_sim=4, params=parameters)
exit()

# Set up parameters for the experiment
exp_parameters = {'agents': ap.IntRange(50, 1000),
                  'number_of_neighbors': ap.IntRange(2, 10),
                  'network_density': ap.Range(0., 1.),
                  'steps': 100000
                  }

sample = ap.Sample(parameters=exp_parameters, n=5)
exp = ap.Experiment(LangChangeModel, sample, iterations=1)
exp_results = exp.run()
exp_results.save()
exit()

# Run N number of simulations
batch_simulate(num_sim=10, params=parameters)
