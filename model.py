# Model design
import random
import agentpy as ap
import numpy as np
import networkx as nx
from utils import batch_simulate
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--ps', type=int, default=10, help='Number of agents')
arg_parser.add_argument('--ifs', type=float, default=0.2, help='Initial frequency of the innovation')
arg_parser.add_argument('--n_ns', type=int, default=4, help='Average number of neighbours per agents')
arg_parser.add_argument('--rp', default=0, help='Rewiring probability: [0, 1.0]')
arg_parser.add_argument('--sp', type=float, default=0.1, help='Selection pressure: [0.1, 1.0]')
arg_parser.add_argument('--nls', default=0.1, help='Proportion of leaders')
arg_parser.add_argument('--sim_steps', default=10000, help='Number of simulation steps')
arg_parser.add_argument('--sim_runs', default=5, help='Number of simulation runs')
arg_parser.add_argument('--m', default='neutral_change', help='Mechanism name')


class Agent(ap.Agent):

    def setup(self) -> None:
        """
        Set up initial states of an agent.
        """

        # Initial distribution of linguistic variants A and B
        self.memory = np.random.choice(self.p.lingueme, size=self.p.memory_size, p=[0, 1])

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
        if self.p.interactor_selection:
            self.p.leaders = int(self.p.agents * self.p.n)
            # Randomly choose the defined number of leaders from the population
            leaders = self.random.sample(self.agents, self.p.leaders)
            # Update the memory and the usage frequency of each agent from the subset
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
    args = arg_parser.parse_args()
    ps = args.ps
    ifs = args.ifs
    ns = args.n_ns
    rp = args.rp
    sp = args.sp
    nls = args.nls
    sim_steps = args.sim_steps
    sim_runs = args.sim_runs

    mechanism = args.m

    parameters = {'agents': ps,
                  'lingueme': ('A', 'B'),
                  'memory_size': 10,
                  'initial_frequency': ifs,
                  'number_of_neighbors': ns,
                  'rewiring_probability': rp,
                  'interactor_selection': mechanism == 'interactor_selection',
                  'replicator_selection': mechanism == 'replicator_selection',
                  'neutral_change': mechanism == 'neutral_change',
                  'selection_pressure': sp,
                  'n': nls,
                  'leaders': None,
                  'steps': sim_steps
                  }
    # print(parameters)

    # Perform and plot a specific number of simulations
    # for one parameter set
    batch_simulate(num_sim=sim_runs, model=LangChangeModel, params=parameters)

