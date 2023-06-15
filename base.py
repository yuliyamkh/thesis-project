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
        This method is called by the model during the setup phase,
        before the simulation starts
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
        Produce an utterance by sampling one token from the memory
        based on the usage frequency x
        """

        self.sampled_token = np.random.choice(self.memory, size=1)[0]

    def reinforce(self) -> None:
        """
        Reinforce the own behaviour replacing a randomly
        removed token in the memory with the sampled
        token's copy
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
        Replacing the randomly removed token in the memory with the
        neighbour's sampled token
        :param neighbour: one of the k agent's neighbours
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

    def update(self):
        """
        Record belief of choosing the innovative variant v1
        based on the updated memory
        """
        self.A = np.count_nonzero(self.memory == 'A') / len(self.memory)


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

    def update(self):
        """
        Record variables after setup and each step
        """

        # Record average frequency of A
        average_A = sum(self.agents.A) / len(self.agents.A)
        self.record('A', average_A)

    def step(self):
        """
        Run interactions according to desired mechanism:
        neutral change, interactor or replicator selection
        """

        for t in range(self.p.time):
            # Choose a random agent from agents
            agent = self.random.choice(self.agents)
            # Initialize neighbors
            neighbors = [j for j in self.network.neighbors(agent)]
            # Select one random neighbor
            neighbor = self.random.choice(neighbors)

            # Perform action
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

        # Report final average frequency of A
        final_average_A = sum(self.agents.A) / len(self.agents.A)
        self.report('final_A', final_average_A)


# Parameters setup
parameters = {'agents': 100,
              'lingueme': ('A', 'B'),
              'memory_size': 10,
              'initial_frequency': 0.5,
              'number_of_neighbors': 4,
              'network_density': 0.01,
              'interactor_selection': False,
              'replicator_selection': True,
              'neutral_change': False,
              'selection_pressure': 0.8,
              'n': 20,
              'time': 100,
              'steps': 1000
              }

batch_simulate(num_sim=15, model=LangChangeModel, params=parameters)
exit()

sample = ap.Sample(parameters=parameters, n=40)
exp = ap.Experiment(LangChangeModel, sample=sample, iterations=3, record=True)
exp_results = exp.run(n_jobs=-1, verbose=10)
exp_results.save()
exit()

