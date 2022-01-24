import numpy as np


class AgentsData:

    def __init__(self, alphas, params, network_containers=None):
        self.alphas = alphas
        self.memory = np.zeros((params.num_agents, params.memory_length))
        self.network_containers = network_containers


class DataModifier:

    def __init__(self, visited_spaces, params):
        self.visited_spaces = visited_spaces
        self.old_counts = np.array([len(x) for x in self.visited_spaces])
        self.update_memory = params.update_memory
        self.counter = 0
        self.sum = np.zeros(params.num_agents)

    def modify(self, agent_data: AgentsData, params):
        self.counter += 1
        if self.update_memory:
            new_counts = np.array([len(x) for x in self.visited_spaces])
            diff = new_counts - self.old_counts
            self.sum += diff
            self.old_counts = new_counts

            if self.counter >= params.memory_compression_factor:
                agent_data.memory[:, :-1] = agent_data.memory[:, 1:]
                agent_data.memory[:, -1] = self.sum
                self.sum = self.sum * 0
                self.counter = 0



    def new_repeat(self, agent_data, params):
        if self.update_memory:
            self.old_counts = np.zeros(params.num_agents)
            agent_data.memory = agent_data.memory * 0
