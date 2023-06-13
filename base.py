# Model design
import random
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

        # Initial distribution of A and B
        self.memory = np.random.choice(self.p.lingueme,
                                       size=self.p.memory_size,
                                       p=[self.p.initial_frequency, 1-self.p.initial_frequency])

        # Frequency of A
        self.A = np.count_nonzero(self.memory == 'A') / len(self.memory)

        # Usage frequency x
        self.x = self.A

        # The produced token
        self.sampled_token = None

    def speak(self) -> None:
        """
        Produce an utterance by sampling one token from the memory
        based on the usage frequency x
        """

        self.sampled_token = np.random.choice(self.p.lingueme, size=1)[0]

    def reinforce(self) -> None:
        """
        Reinforce the own behaviour replacing a randomly
        removed token in the memory with the sampled
        token's copy
        """
        if self.p.neutral_change:
            # Choose a random index to remove
            random_index = np.random.randint(len(self.memory))

            # Remove the element at the random index
            self.memory = np.delete(self.memory, random_index)

            # Append the sampled token
            self.memory = np.append(self.memory, self.sampled_token)

        if self.p.replicator_selection and self.p.interactor_selection:
            if self.id < self.p.n:
                if self.sampled_token == 'A':
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Remove the element at the random index
                        self.memory = np.delete(self.memory, random_index)
                        # Append the neighbour's sampled token
                        self.memory = np.append(self.memory, self.sampled_token)

        else:
            if self.p.replicator_selection:
                if self.sampled_token == 'A':
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Remove the element at the random index
                        self.memory = np.delete(self.memory, random_index)
                        # Append the neighbour's sampled token
                        self.memory = np.append(self.memory, self.sampled_token)

            if self.p.interactor_selection:
                if self.id < self.p.n:
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Remove the element at the random index
                        self.memory = np.delete(self.memory, random_index)
                        # Append the neighbour's sampled token
                        self.memory = np.append(self.memory, self.sampled_token)

    def listen(self, neighbour) -> None:
        """
        Listen to the neighbour to match more closely his behaviour
        Replacing the randomly removed token in the memory with the
        neighbour's sampled token
        :param neighbour: one of the k agent's neighbours
        """

        if self.p.neutral_change:
            # Choose a random index to remove
            random_index = np.random.randint(len(self.memory))
            # Remove the element at the random index
            self.memory = np.delete(self.memory, random_index)
            # Append the neighbour's sampled token
            self.memory = np.append(self.memory, neighbour.sampled_token)

        if self.p.replicator_selection and self.p.interactor_selection:
            if self.id > self.p.n:
                if neighbour.id <= self.p.n:
                    if self.sampled_token == 'B' and neighbour.sampled_token == 'A':
                        if random.random() < self.p.selection_pressure:
                            # Choose a random index to remove
                            random_index = np.random.randint(len(self.memory))
                            # Remove the element at the random index
                            self.memory = np.delete(self.memory, random_index)
                            # Append the neighbour's sampled token
                            self.memory = np.append(self.memory, neighbour.sampled_token)
        else:
            if self.p.interactor_selection_2:
                if self.id > self.p.n:
                    if neighbour.id <= self.p.n:
                        if random.random() < self.p.selection_pressure:
                            # Choose a random index to remove
                            random_index = np.random.randint(len(self.memory))
                            # Remove the element at the random index
                            self.memory = np.delete(self.memory, random_index)
                            # Append the neighbour's sampled token
                            self.memory = np.append(self.memory, neighbour.sampled_token)

            if self.p.interactor_selection:
                if self.id > self.p.n:
                    if neighbour.id <= self.p.n:
                        if random.random() < self.p.selection_pressure:
                            # Choose a random index to remove
                            random_index = np.random.randint(len(self.memory))
                            # Remove the element at the random index
                            self.memory = np.delete(self.memory, random_index)
                            # Append the neighbour's sampled token
                            self.memory = np.append(self.memory, neighbour.sampled_token)

            if self.p.replicator_selection:
                if self.sampled_token == 'B' and neighbour.sampled_token == 'A':
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Remove the element at the random index
                        self.memory = np.delete(self.memory, random_index)
                        # Append the neighbour's sampled token
                        self.memory = np.append(self.memory, neighbour.sampled_token)

    def update(self):
        """
        Record belief of choosing the innovative variant v1
        based on the updated memory
        """
        self.A = np.count_nonzero(self.memory == 'A') / len(self.memory)
        self.x = self.A


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
        # Mechanism: neutral change
        self.agents = ap.AgentList(self, self.p.agents, Agent)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

        # Change setup of agents
        # Mechanism: interactor selection
        if self.p.interactor_selection:
            for agent in self.agents:
                if agent.id <= self.p.n:
                    agent.x = 1
                    agent.memory = np.random.choice(self.p.lingueme,
                                                    size=self.p.memory_size,
                                                    p=[agent.x, 1-agent.x])
                if agent.id > self.p.n:
                    agent.x = 0
                    agent.memory = np.random.choice(self.p.lingueme,
                                                    size=self.p.memory_size,
                                                    p=[agent.x, 1-agent.x])

    def action(self, agent, neighbor) -> None:
        """
        Definition of actions performed by agent and
        its neighbor during one interaction
        :param agent: agent
        :param neighbor: neighbour
        :return: None
        """

        agent.speak()
        neighbor.speak()

        agent.reinforce()
        neighbor.reinforce()

        agent.listen(neighbor)
        neighbor.listen(agent)

        agent.update()
        neighbor.update()

    def run_interactions(self, network):
        """
        Run interactions between agents and their neighbours.
        Choose two agents who are in a neighborhood
        to each other to interact and perform the actions
        of speaking, reinforcing, and listening
        :return: None
        """

        # Choose a random agent from agents
        agent = self.random.choice(network.agents.to_list())

        # Initialize neighbors
        neighbors = [j for j in network.neighbors(agent)]

        # Select one random neighbor
        neighbor = self.random.choice(neighbors)

        # Perform action
        self.action(agent=agent, neighbor=neighbor)

    def update(self):
        """
        Record variables after setup and each step
        """

        # Record frequency of A
        freq_a = sum(self.network.agents.A) / len(self.network.agents.A)

        x = sum(self.network.agents.x) / len(self.network.agents.x)

        # Record the data using self.record()
        self.record('A', freq_a)
        self.record('x', x)

    def step(self):
        """
        Run interactions according to desired mechanism:
        neutral change, interactor or replicator selection
        """

        for t in range(self.p.time):
            self.run_interactions(self.network)

    def end(self):
        """
        Record evaluation measures at the end of the simulation.
        """
        final_A = sum(self.network.agents.A) / len(self.network.agents.A)
        self.report('final_x', final_A)


# Set up parameters for the model
parameters = {'agents': ap.IntRange(100, 1500),
              'lingueme': ('A', 'B'),
              'memory_size': 10,
              'initial_frequency': 0.2,
              'number_of_neighbors': 8,
              'network_density': 0.01,
              'interactor_selection': False,
              'replicator_selection': False,
              'neutral_change': False,
              'interactor_selection_2': True,
              'selection_pressure': 0.8,
              'n': 20,
              'time': 100,
              'steps': 1000
              }

sample = ap.Sample(parameters=parameters, n=40)
exp = ap.Experiment(LangChangeModel, sample=sample, iterations=3, record=True)
exp_results = exp.run(n_jobs=-1, verbose=10)
exp_results.save()
exit()

batch_simulate(num_sim=5, model=LangChangeModel, params=parameters)
