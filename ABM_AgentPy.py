# Model design
import copy
import agentpy as ap
import numpy as np
import networkx as nx
from utils import replicator_selection
from utils import interactor_selection
from utils import batch_simulate
import random


class Agent(ap.Agent):

    def setup(self) -> None:
        """
        Set up an agent's initial states.
        This method is called by the model during the setup phase,
        before the simulation starts
        """

        self.memory = []

        if self.id <= self.p.n:
            self.memory = np.random.choice(self.p.lingueme, size=self.p.memory_size, p=[1, 0])
        if self.id > self.p.n:
            self.memory = np.random.choice(self.p.lingueme, size=self.p.memory_size, p=[0, 1])

        # Initial distribution of v1 and v2
        # self.memory = np.random.choice(self.p.lingueme, size=self.p.memory_size, p=[self.p.initial_frequency, 1-self.p.initial_frequency])

        # Probability of choosing v1
        self.x = np.count_nonzero(self.memory == 'v1') / len(self.memory)

        # The copy of the initial memory will be updated during interactions
        self.updated_memory = copy.deepcopy(self.memory)

        # The copy of the initial belief will be updated during interactions
        self.updated_x = copy.deepcopy(self.x)

        # The produced utterance
        self.sampled_token = ''

    def speak(self) -> None:
        """
        Produce an utterance by sampling one token from the store
        """

        # if replicator_select:
        #     self.sampled_token = np.random.choice(self.p.lingueme, size=1, p=[self.x, 1-self.x])[0]
        self.sampled_token = np.random.choice(self.p.lingueme, size=1, p=[self.updated_x, 1 - self.updated_x])[0]

    def reinforce(self) -> None:
        """
        Reinforce the own behaviour replacing a randomly
        removed token in the store with the sampled
        token's copy
        """

        # Choose a random index to remove
        random_index = np.random.randint(len(self.updated_memory))

        # Remove the element at the random index
        self.updated_memory = np.delete(self.updated_memory, random_index)

        # Append the sampled token
        self.updated_memory = np.append(self.updated_memory, self.sampled_token)

        # Selection
        # if replicator_select:
        #     if self.sampled_token == 'v1':
        #         self.x = replicator_selection(self.x, b=0.001)

    def listen(self, neighbour) -> None:
        """
        Listen to the neighbour to match more closely his behaviour
        Replacing the randomly removed token in the story with the
        neighbour's sampled token
        :param select: True or False
        :param neighbour: one of the k agent's neighbours
        """

        # Replicator selection
        # if neighbour.sampled_token == 'v1':
        #     self.x = replicator_selection(self.x, b=0.001)

        # Interactor selection
        if self.id > self.p.n and neighbour.id <= self.p.n:
            # Choose a random index to remove
            randon_index = np.random.randint(len(self.updated_memory))
            # Remove the element at the random index
            self.updated_memory = np.delete(self.updated_memory, randon_index)
            # Append the neighbour's sampled token
            self.updated_memory = np.append(self.updated_memory, neighbour.sampled_token)

    def update(self):
        """
        Record belief of choosing the innovation variant v1
        based on the updated store.
        """
        self.updated_x = np.count_nonzero(self.updated_memory == 'v1') / len(self.updated_memory)
        self.record('x', self.updated_x)

        # if select:
        #     self.record('initial_x', self.x)


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
        average_updated_x = sum(self.agents.updated_x) / len(self.agents.updated_x)
        # average_x = sum(self.agents.x) / len(self.agents.x)
        # self.record('average_x', average_x)
        self.record('average_updated_x', average_updated_x)

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
        final_average_updated_x = sum(self.agents.updated_x) / len(self.agents.updated_x)
        final_average_x = sum(self.agents.x) / len(self.agents.x)
        self.report('Final_average_updated_x', final_average_updated_x)
        self.report('Final_average_x', final_average_x)


# Set up parameters for the model
parameters = {'agents': 100,
              'n': 5,
              'a': 50,
              'lingueme': ['v1', 'v2'],
              'memory_size': 10,
              'initial_frequency': 0.01,
              'number_of_neighbors': 8,
              'network_density': 0.01,
              'steps': 10000
              }

# model = LangChangeModel()
# results = model.run()
# print(results.save())
# exit()
# Run N number of simulations
random.seed(123)
batch_simulate(num_sim=2, model=LangChangeModel, params=parameters)
exit()

# Set up parameters for the experiment
exp_parameters = {'agents': 100000,
                  'lingueme': ['v1', 'v2'],
                  'memory_size': 10,
                  'initial_frequency': 0.01,
                  'number_of_neighbors': 8,
                  'rewiring_probability': 0,
                  'steps': 500000
                  }

sample = ap.Sample(parameters=exp_parameters)
exp = ap.Experiment(LangChangeModel, sample, iterations=3, record=True)
exp_results = exp.run()
exp_results.save()
exit()
