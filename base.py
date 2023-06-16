# Model design
import random
import agentpy as ap
import numpy as np
import networkx as nx
from utils import batch_simulate


class Agent(ap.Agent):

    def setup(self) -> None:
        """
        Set up initial states of an agent.
        """

        # Initial distribution of linguistic variants A and B
        self.memory = np.random.choice(self.p.lingueme,
                                       size=self.p.memory_size,
                                       p=[self.p.initial_frequency, 1-self.p.initial_frequency])

        # Frequency of A
        self.A = np.count_nonzero(self.memory == 'A') / len(self.memory)

        # The produced token
        self.sampled_token = None

    def speak(self) -> None:
        """
        Produce an utterance by randomly
        sampling one token from the memory
        """

        self.sampled_token = np.random.choice(self.memory, size=1)[0]

    def reinforce(self) -> None:
        """
        Reinforce the own behaviour replacing a randomly
        removed token in the memory with the copy of
        the sampled token

        The reinforcement process is preformed according
        to the three mechanisms of language change:
        1. neutral change; 2. interactor selection, and
        3. replicator selection
        """

        # Neutral mechanism
        if self.p.neutral_change:
            # Choose a random index to remove
            random_index = np.random.randint(len(self.memory))
            # Replace with the sampled token
            self.memory[random_index] = self.sampled_token

        # Replicator selection and interactor selection mechanisms
        if self.p.replicator_selection and self.p.interactor_selection:
            if self.id < self.p.n:
                if self.sampled_token == 'A':
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Insert the sampled token
                        self.memory[random_index] = self.sampled_token

        else:
            # Replicator selection
            if self.p.replicator_selection:
                if self.sampled_token == 'A':
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Replace with the sampled token
                        self.memory[random_index] = self.sampled_token

            # Interactor selection
            if self.p.interactor_selection:
                if self.id < self.p.n:
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Replace with the token sampled by the neighbour
                        self.memory[random_index] = self.sampled_token

    def listen(self, neighbour) -> None:
        """
        Listen to the neighbour to match more closely his behaviour
        by replacing the randomly removed token in the memory
        with the token sampled by the neighbour

        The process of listening is preformed according
        to the three mechanisms of language change:
        1. neutral change; 2. interactor selection, and
        3. replicator selection
        :param neighbour: one of the k neighbours of an agent
        """

        # Neutral mechanism
        if self.p.neutral_change:
            # Choose a random index to remove
            random_index = np.random.randint(len(self.memory))
            # Replace with the token sampled by the neighbour
            self.memory[random_index] = neighbour.sampled_token

        # Replicator selection and interactor selection
        if self.p.replicator_selection and self.p.interactor_selection:
            if self.id > self.p.n:
                if neighbour.id <= self.p.n:
                    if neighbour.sampled_token == 'A':
                        if random.random() < self.p.selection_pressure:
                            # Choose a random index to remove
                            random_index = np.random.randint(len(self.memory))
                            # Replace with the token sampled by the neighbour
                            self.memory[random_index] = neighbour.sampled_token
        else:
            # Interactor selection
            if self.p.interactor_selection:
                if self.id > self.p.n:
                    if neighbour.id <= self.p.n:
                        if random.random() < self.p.selection_pressure:
                            # Choose a random index to remove
                            random_index = np.random.randint(len(self.memory))
                            # Replace with the token sampled by the neighbour
                            self.memory[random_index] = neighbour.sampled_token

            # Replicator selection
            if self.p.replicator_selection:
                if neighbour.sampled_token == 'A':
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Replace with the token sampled by the neighbour
                        self.memory[random_index] = neighbour.sampled_token

    def update(self) -> None:
        """
        Record the frequency of the innovative variant A
        based on the updated memory
        """
        self.A = np.count_nonzero(self.memory == 'A') / len(self.memory)


class LangChangeModel(ap.Model):

    def setup(self) -> None:
        """
        Initialize a population of agents and
        the network in which they exist and interact
        """

        # Initialize a graph
        graph = nx.watts_strogatz_graph(
            self.p.agents,
            self.p.number_of_neighbors,
            self.p.rewiring_probability
        )

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Agent)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

    def update(self) -> None:
        """
        Record the average frequency of the innovative
        variant A after setup and each step
        """

        average_A = sum(self.agents.A) / len(self.agents.A)
        self.record('A', average_A)

    def step(self) -> None:
        """
        Before the interaction starts, one agent is randomly
        sampled from the network. Next, another agent, the
        interlocutor, is randomly sampled from its neighborhood.

        During the interaction, each of the two agents produces
        a linguistic variant A or B, reinforces its own behaviour,
        copies the behaviour of the neighbour, and updates its
        frequency of using the innovative variant A based on its
        updated memory
        """

        # Choose a random agent from agents
        agent = self.random.choice(self.agents)
        # Initialize neighbors
        neighbors = [j for j in self.network.neighbors(agent)]
        # Select one random neighbor
        neighbor = self.random.choice(neighbors)

        # Perform actions during the interaction
        agent.speak()
        neighbor.speak()

        agent.reinforce()
        neighbor.reinforce()

        agent.listen(neighbor)
        neighbor.listen(agent)

        agent.update()
        neighbor.update()

    def end(self) -> None:
        """
        Report final average frequency of A
        at the end of the simulation.
        """

        final_average_A = sum(self.agents.A) / len(self.agents.A)
        self.report('final_A', final_average_A)


if __name__ == '__main__':

    # Parameters setup
    parameters = {'agents': 10,
                  'lingueme': ('A', 'B'),
                  'memory_size': 10,
                  'initial_frequency': 0.2,
                  'number_of_neighbors': 4,
                  'rewiring_probability': 0.01,
                  'interactor_selection': False,
                  'replicator_selection': False,
                  'neutral_change': True,
                  'selection_pressure': 0.8,
                  'n': 2,
                  'steps': 1000
                  }

    # Perform and plot a specific number of simulations
    # for one parameter set
    batch_simulate(num_sim=3, model=LangChangeModel, params=parameters)

