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

        graph = nx.watts_strogatz_graph(
            self.p.agents,
            self.p.number_of_neighbors,
            self.p.network_density
        )

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.agents, Agent)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

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

    def update(self):
        """
        Record variables after setup and each step
        """

        # Record average probability x after each simulation step
        average_updated_x = sum(self.agents.updated_x) / len(self.agents.updated_x)
        self.record('x', average_updated_x)

    def step(self):
        """
        Choose two agents who are in a neighborhood
        to each other to interact and perform the actions
        of speaking, reinforcing, and listening
        """

        for t in range(self.p.time):
            # Choose a random agent from agents
            agent = self.random.choice(self.agents)

            # Initialize neighbors
            neighbors = [j for j in self.network.neighbors(agent)]

            # Select one random neighbor
            neighbor = self.random.choice(neighbors)

            # Perform action
            self.action(agent=agent, neighbor=neighbor)

    def end(self):
        """
        Record evaluation measures at the end of the simulation.
        """
        final_average_updated_x = sum(self.agents.updated_x) / len(self.agents.updated_x)
        self.report('final_x', final_average_updated_x)


# Set up parameters for the model
parameters = {'agents': ap.IntRange(50, 10000),
              'lingueme': ('v1', 'v2'),
              'memory_size': 10,
              'initial_frequency': 0.2,
              'number_of_neighbors': 2,
              'network_density': 0.01,
              'time': 500,
              'steps': 3000
              }

sample = ap.Sample(parameters=parameters, n=10)
exp = ap.Experiment(LangChangeModel, sample=sample, iterations=3, record=True)
exp_results = exp.run(n_jobs=-1, verbose=10)
exp_results.save()

exit()
batch_simulate(num_sim=3, model=LangChangeModel, params=parameters)
