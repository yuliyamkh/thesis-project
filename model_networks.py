# Model design
import copy
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networks_split import split_population


class Agent(ap.Agent):

    def setup(self) -> None:
        """
        Set up an agent's initial states.
        This method is called by the model during the setup phase,
        before the simulation starts
        """

        # Initial distribution of v1 and v2
        self.memory = np.random.choice(self.p.lingueme,
                                       size=self.p.memory_size,
                                       p=[self.p.initial_frequency, 1-self.p.initial_frequency])

        # Probability of choosing the innovative variant v1
        self.x = self.p.initial_frequency

        # Updated probability of choosing the innovative variant v1
        self.updated_x = copy.deepcopy(self.x)

        # The produced token
        self.sampled_token = None

        # Initialize name of agent
        self.name = None

    def speak(self) -> None:
        """
        Produce an utterance by sampling one token from the memory
        based on the usage frequency x
        """

        self.sampled_token = np.random.choice(self.p.lingueme,
                                              size=1,
                                              p=[self.updated_x, 1 - self.updated_x])[0]

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
        :param interactor_selection: True or False
        :param neighbour: one of the k agent's neighbours
        """

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
        self.updated_x = np.count_nonzero(self.memory == 'v1') / len(self.memory)


class LangChangeModel(ap.Model):

    def setup(self) -> None:
        """
        Initialize a population of agents and
        the network in which they exist and interact
        """

        self.split = 0
        self.interaction = 0
        self.iteration = 0

        self.subgraphs = []
        self.subgraph_networks = []
        self.subgraphs2 = {}
        self.subgraph_networks2 = {}

        graph = nx.watts_strogatz_graph(
            self.p.agents,
            self.p.number_of_neighbors,
            self.p.network_density
        )

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Agent)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

    def interact(self, network):
        """Agents interaction"""
        agent = self.random.choice(list(network.agents))
        neighbors = [j for j in network.neighbors(agent)]
        neighbor = self.random.choice(neighbors)

        agent.speak()
        neighbor.speak()

        agent.reinforce()
        neighbor.reinforce()

        agent.listen(neighbor)
        neighbor.listen(agent)

        agent.update()
        neighbor.update()

    def split_network(self):
        """Split network"""

        self.subgraphs = split_population(self.network.graph)

        for subgraph in self.subgraphs:
            self.subgraph_network = ap.Network(self, subgraph)
            self.subgraph_network.add_agents(self.agents, self.subgraph_network.nodes)
            self.subgraph_networks.append(self.subgraph_network)

    def split_network2(self):
        """Split network 2"""
        for i, subgraph_network in enumerate(self.subgraph_networks):
            self.subgraph_networks2[i] = []
            self.subgraphs = split_population(subgraph_network.graph)
            for subgraph in self.subgraphs:
                self.subgraph_network = ap.Network(self, subgraph)
                self.subgraph_network.add_agents(self.agents, self.subgraph_network.nodes)
                self.subgraph_networks2[i].append(self.subgraph_network)

    def split_network3(self):
        """Split network 3"""
        for i, subgraph_networks in self.subgraph_networks2.items():
            self.subgraph_networks2[i] = [[] for _ in range(len(subgraph_networks))]
            for idx in range(len(subgraph_networks)):
                graph = subgraph_networks[idx].graph
                self.subgraphs = split_population(graph)
                for subgraph in self.subgraphs:
                    self.subgraph_network = ap.Network(self, subgraph)
                    self.subgraph_network.add_agents(self.agents, self.subgraph_network.nodes)
                    self.subgraph_networks2[i][idx].append(self.subgraph_network)

    def update(self):
        """
        Record variables after setup and each step
        """

        if self.interaction <= 3:
            # Record average probability x after each simulation step
            average_updated_x = sum(self.agents.updated_x) / len(self.agents.updated_x)
            self.record('x', average_updated_x)

        if 3 < self.interaction <= 6:
            self.res = [[] for _ in range(len(self.subgraph_networks))]
            for i, subgraph in enumerate(self.subgraph_networks):
                i, res = i, sum(subgraph.agents.updated_x) / len(subgraph.agents.updated_x)
                self.res[i] = res
            self.record('x', self.res)

        if 6 < self.interaction <= 9:
            self.res = [[] for _ in range(len(list(self.subgraph_networks2.keys())))]
            for i, subgraphs in self.subgraph_networks2.items():
                for idx in range(len(subgraphs)):
                    i, res = i, sum(subgraphs[idx].agents.updated_x) / len(subgraphs[idx].agents.updated_x)
                    self.res[i].append(res)
            self.record('x', self.res)

        else:
            result = []
            for i, subgraphs in self.subgraph_networks2.items():
                if subgraphs:
                    self.res = [[None for _ in inner] for inner in subgraphs]
                    for idx in range(len(subgraphs)):
                        for sg in range(len(subgraphs[idx])):
                            sg, res = sg, sum(subgraphs[idx][sg].agents.updated_x) / len(subgraphs[idx][sg].agents.updated_x)
                            self.res[idx][sg] = res
                    result.append(self.res)
            self.record('x', result)

    def step(self):
        """
        Choose two agents who are in a neighborhood
        to each other to interact and perform the actions
        of speaking, reinforcing, and listening
        """

        if self.iteration == 3:
            if self.split == 0:
                # Network split
                self.split_network()
                self.iteration = 0
                self.split += 1

        if self.iteration == 3:
            if self.split == 1:
                self.split_network2()
                self.iteration = 0
                self.split += 1

        if self.iteration == 3:
            if self.split == 2:
                self.split_network3()
                self.iteration = 0
                self.split += 1

        if self.iteration == 3:
            if self.split >= 3:
                self.split_network3()
                self.iteration = 0
                self.split += 1

        if self.interaction >= 9 and self.split >= 3:
            self.interaction += 1
            self.iteration += 1
            for i, subgraphs in self.subgraph_networks2.items():
                if subgraphs:
                    for idx in range(len(subgraphs)):
                        for sg in range(len(subgraphs[idx])):
                            agent = self.random.choice(list(subgraphs[idx][sg].agents))
                            neighbors = [j for j in subgraphs[idx][sg].neighbors(agent)]
                            neighbor = self.random.choice(neighbors)

                            agent.speak()
                            neighbor.speak()

                            agent.reinforce()
                            neighbor.reinforce()

                            agent.listen(neighbor)
                            neighbor.listen(agent)

                            agent.update()
                            neighbor.update()

        if 6 <= self.interaction < 9 and self.split == 2:
            self.interaction += 1
            self.iteration += 1
            for i, subgraphs in self.subgraph_networks2.items():
                for idx in range(len(subgraphs)):
                    agent = self.random.choice(list(subgraphs[idx].agents))
                    neighbors = [j for j in subgraphs[idx].neighbors(agent)]
                    neighbor = self.random.choice(neighbors)

                    agent.speak()
                    neighbor.speak()

                    agent.reinforce()
                    neighbor.reinforce()

                    agent.listen(neighbor)
                    neighbor.listen(agent)

                    agent.update()
                    neighbor.update()

        if 3 <= self.interaction < 6 and self.split == 1:
            self.interaction += 1
            self.iteration += 1
            for i, subgraph in enumerate(self.subgraph_networks):
                agent = self.random.choice(list(subgraph.agents))
                neighbors = [j for j in subgraph.neighbors(agent)]
                neighbor = self.random.choice(neighbors)

                agent.speak()
                neighbor.speak()

                agent.reinforce()
                neighbor.reinforce()

                agent.listen(neighbor)
                neighbor.listen(agent)

                agent.update()
                neighbor.update()

        if self.interaction < 3:
            self.interaction += 1
            self.iteration += 1

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
        pass
        # final_average_updated_x = sum(self.agents.updated_x) / len(self.agents.updated_x)
        # self.report('Final_average_updated_x', final_average_updated_x)


# Set up parameters for the model
parameters = {'agents': 100,
              'lingueme': ['v1', 'v2'],
              'memory_size': 10,
              'initial_frequency': 0.4,
              'number_of_neighbors': 8,
              'network_density': 0.01,
              'steps': 100
              }

model = LangChangeModel(parameters)
results = model.run()
print(results.save())
print(results.variables.LangChangeModel)