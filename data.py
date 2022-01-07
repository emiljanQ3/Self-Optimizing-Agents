import numpy as np


class AgentsData:

    def __init__(self, alphas, params):
        self.alphas = alphas
        self.memory = np.zeros((params.num_agents, params.memory_length))


class DataModifier:

    def __init__(self, visited_spaces, params):
        self.visited_spaces = visited_spaces
        self.old_counts = np.array([len(x) for x in self.visited_spaces])
        self.update_memory = params.update_memory

    def modify(self, agent_data: AgentsData, params):
        if self.update_memory:
            new_counts = np.array([len(x) for x in self.visited_spaces])
            diff = new_counts - self.old_counts

            agent_data.memory[:, :-1] = agent_data.memory[:, 1:]
            agent_data.memory[:, -1] = diff

            self.old_counts = new_counts

    def new_repeat(self, agent_data, params):
        if self.update_memory:
            self.old_counts = np.zeros(params.num_agents)
            agent_data.memory = agent_data.memory * 0
