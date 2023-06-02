# Model design
import copy
import agentpy as ap
import numpy as np
import networkx as nx
from utils import batch_simulate


class Agent(ap.Agent):

    def setup(self) -> None:
        """
        Set up an agent's initial states.
        This method is called by the model during the setup phase,
        before the simulation starts
        """

        # Initialize memory
        self.memory = []

        # Initialize initial probability of choosing innovative variant v1
        self.x = 0

        if self.id <= self.p.n:
            self.memory = np.random.choice(self.p.lingueme,
                                           size=self.p.memory_size,
                                           p=[1, 0])
            self.x = 1

        if self.id > self.p.n:
            self.memory = np.random.choice(self.p.lingueme,
                                           size=self.p.memory_size,
                                           p=[0, 1])
            self.x = 0

        # The produced token
        self.sampled_token = None

    def speak(self) -> None:
        """
        Produce an utterance by sampling one token from the memory
        based on the usage frequency x
        """

        self.sampled_token = np.random.choice(self.p.lingueme, size=1, p=[self.x, 1 - self.x])[0]

    def reinforce(self) -> None:
        """
        Reinforce the own behaviour replacing a randomly
        removed token in the memory with the sampled
        token's copy
        """

        # Choose a random index to remove
        random_index = np.random.randint(len(self.memory))

        # Remove the element at the random index
        self.memory = np.delete(self.memory, random_index)

        # Append the sampled token
        self.memory = np.append(self.memory, self.sampled_token)

    def listen(self, neighbour) -> None:
        """
        Listen to the neighbour to match more closely his behaviour
        Replacing the randomly removed token in the memory with the
        neighbour's sampled token
        :param neighbour: one of the k agent's neighbours
        """

        if self.id > self.p.n >= neighbour.id:
            # Choose a random index to remove
            random_index = np.random.randint(0, len(self.memory)-1)
            # Remove the element at the random index
            self.memory = np.delete(self.memory, random_index)
            # Append the neighbour's sampled token
            self.memory = np.append(self.memory, neighbour.sampled_token)

    def update(self):
        """
        Record belief of choosing the innovative variant v1
        based on the updated memory
        """
        self.x = np.count_nonzero(self.memory == 'v1') / len(self.memory)


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

        # Record average probability x after each simulation step
        average_x = sum(self.agents.x) / len(self.agents.x)
        self.record('average_updated_x', average_x)

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
        final_average_x = sum(self.agents.x) / len(self.agents.x)
        self.report('Final_average_updated_x', final_average_x)


# Set up parameters for the model
parameters = {'agents': 100,
              'lingueme': ['v1', 'v2'],
              'memory_size': 10,
              'n': 10,
              'number_of_neighbors': 20,
              'network_density': 0,
              'steps': 100000
              }

# model = LangChangeModel(parameters)
# results = model.run()
# Run N number of simulations
batch_simulate(num_sim=15, model=LangChangeModel, params=parameters)