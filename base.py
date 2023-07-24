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
                                       p=[0, 1])

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
        """

        # Choose a random index to remove
        random_index = np.random.randint(len(self.memory))
        # Replace with the sampled token
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

        # Interactor selection
        if self.p.interactor_selection:
            if self.id > self.p.leaders:
                if neighbour.id <= self.p.leaders:
                    if random.random() < self.p.selection_pressure:
                        # Choose a random index to remove
                        random_index = np.random.randint(len(self.memory))
                        # Replace with the token sampled by the neighbour
                        self.memory[random_index] = neighbour.sampled_token

        # Replicator selection
        if self.p.replicator_selection:
            if self.sampled_token == 'B' and neighbour.sampled_token == 'A':
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

        # Change setup of agents according to the mechanism operating

        # Mechanisms: Neutral change and replicator selection
        if self.p.replicator_selection or self.p.neutral_change:
            # Compute the number of agents who use an innovation
            num_innovation_agents = int(self.p.initial_frequency * self.p.agents)
            # Randomly choose the defined number of agents from the population
            innovation_agents = self.random.sample(self.agents, num_innovation_agents)
            # Update the memory and the usage frequency of each agent from the subset
            for agent in innovation_agents:
                agent.memory = np.random.choice(self.p.lingueme, size=self.p.memory_size, p=[1, 0])
                agent.A = np.count_nonzero(agent.memory == 'A') / len(agent.memory)

        # Mechanism: Interactor selection
        # Compute the number of leaders
        self.p.leaders = int(self.p.agents * self.p.n)
        # Randomly choose the defined number of leaders from the population
        leaders = self.random.sample(self.agents, self.p.leaders)
        # Update the memory and the usage frequency of each agent from the subset
        if self.p.interactor_selection:
            for agent in leaders:
                agent.memory = np.random.choice(self.p.lingueme, size=self.p.memory_size, p=[1, 0])
                agent.A = np.count_nonzero(agent.memory == 'A') / len(agent.memory)

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
                  'initial_frequency': 0.5,
                  'number_of_neighbors': 4,
                  'rewiring_probability': 0.01,
                  'interactor_selection': True,
                  'replicator_selection': False,
                  'neutral_change': False,
                  'selection_pressure': 0.3,
                  'n': 0.2,
                  'leaders': None,
                  'steps': 1000
                  }

    # Perform and plot a specific number of simulations
    # for one parameter set
    batch_simulate(num_sim=10, model=LangChangeModel, params=parameters)

