import copy

import numpy as np
from tags import ResultTag
from config import Params
from data import AgentsData


class DataRecorder:
    def __init__(self, components):
        self.components = components

    def record(self, agents, new_agents, agent_data, world, mover, rep, step):
        for c in self.components:
            c.record(agents, new_agents, agent_data, world, mover, rep, step)

    def new_repeat(self):
        for c in self.components:
            c.new_repeat()

    def get_results(self):
        results = dict()
        for c in self.components:
            results[c.tag] = c.get_results()

        return results


class PositionRecorder:
    def __init__(self, params):
        self.positions_over_time = -np.ones([params.num_repeats, params.num_steps, params.num_agents, 2])
        self.tag = ResultTag.POSITION

    def record(self, agents, new_agents, agent_data, world, mover, rep, step):
        self.positions_over_time[rep, step] = agents[:, :2]

    def new_repeat(self):
        self.record = lambda *args: ()

    def get_results(self):
        return self.positions_over_time


class AreaGridRecorder:
    def __init__(self, params):
        self.tag = ResultTag.AREA
        self.visited_spaces = [set() for i in range(params.num_agents)]
        self.area_unit_size = params.area_unit_size
        self.saved_area = []

    def new_repeat(self):
        self.saved_area.append([len(x)*self.area_unit_size**2 for x in self.visited_spaces])
        [x.clear() for x in self.visited_spaces]

    def record(self, agents, new_agents, agent_data, world, mover, rep, step):
        for i in range(np.shape(agents)[0]):
            area_unit_pos = np.floor(agents[i, :2] / self.area_unit_size)
            int_pos = (int(area_unit_pos[0]), int(area_unit_pos[1]))
            self.visited_spaces[i].add(int_pos)

    def get_results(self):
        return self.saved_area


class AreaOverTimeRecorder:
    def __init__(self, params):
        self.tag = ResultTag.AREA_TIME
        self.visited_spaces = [dict() for i in range(params.num_agents)]
        self.area_unit_size = params.area_unit_size
        self.saved_area = []

    def new_repeat(self):
        self.saved_area.extend(copy.deepcopy(self.visited_spaces))
        [x.clear() for x in self.visited_spaces]

    def record(self, agents, new_agents, agent_data, world, mover, rep, step):
        for i in range(np.shape(agents)[0]):
            area_unit_pos = np.floor(agents[i, :2] / self.area_unit_size)
            int_pos = (int(area_unit_pos[0]), int(area_unit_pos[1]))
            if int_pos not in self.visited_spaces[i]:
                self.visited_spaces[i][int_pos] = step

    def get_results(self):
        results = []
        for d in self.saved_area:
            steps = []
            for key in d:
                steps.append(d[key])
            results.append(steps)

        return results


class AreaIndexRecorder:
    def __init__(self, params):
        self.tag = ResultTag.AREA_INDICES
        self.visited_spaces = [set() for i in range(params.num_agents)]
        self.area_unit_size = params.area_unit_size

    def record(self, agents, new_agents, agent_data, world, mover, rep, step):
        for i in range(np.shape(agents)[0]):
            area_unit_pos = np.floor(agents[i, :2] / self.area_unit_size)
            int_pos = (int(area_unit_pos[0]), int(area_unit_pos[1]))
            self.visited_spaces[i].add(int_pos)

    def new_repeat(self):
        self.record = lambda *args: ()

    def get_results(self):
        return self.visited_spaces


class ParamRecorder:
    def __init__(self, params):
        self.tag = ResultTag.PARAM
        self.params = copy.deepcopy(params)

    def new_repeat(self):
        pass

    def record(self, agents, new_agents, agent_data, world, mover, rep, step):
        pass

    def get_results(self):
        return self.params


class LossRecorder:
    def __init__(self, params):
        self.num_agents = params.num_agents
        self.tag = ResultTag.LOSS
        self.losses = [[] for _ in range(params.num_agents*params.num_repeats)]
        self.repeat_index = 0

    def new_repeat(self):
        self.repeat_index += 1

    def record(self, agents, new_agents, agent_data: AgentsData, world, mover, rep, step):
        for i in range(self.num_agents):
            j = i + self.repeat_index * self.num_agents
            self.losses[j].append(agent_data.network_containers[j].last_loss)

    def get_results(self):
        return self.losses


class ActionBufferRecorder:
    def __init__(self, params):
        self.num_agents = params.num_agents
        self.tag = ResultTag.BUFFER
        self.buffer = []
        self.last_state = [None for _ in range(self.num_agents)]
        self.last_direction = np.zeros(self.num_agents)

    def new_repeat(self):
        self.last_state = [None for _ in range(self.num_agents)]
        self.last_direction = np.zeros(self.num_agents)

    def record(self, agents, new_agents, agents_data: AgentsData, world, mover, rep, step):
        turning_idx = np.nonzero(self.last_direction != agents[:, 2])
        for i in turning_idx:
            mean_reward = agents_data.reward_since_last_action[i] / agents_data.steps_since_last_action[i]
            if self.last_state[i] is not None:
                self.buffer.append((self.last_state[i], agents_data.alphas[i], mean_reward))

            self.last_state[i] = agents_data.memory[i]
            agents_data.reward_since_last_action[i] = 0
            agents_data.steps_since_last_action[i] = 0

        self.last_direction = agents[:, 2]

    def get_results(self):
        return self.buffer


# class NetworkSaver:
#     def __init__(self, params):
#         self.tag = None
#         self.params = copy.deepcopy(params)
#
#     def new_repeat(self):
#         pass
#
#     def record(self, agents, new_agents, agent_data, world, mover, rep, step):
#         pass
#
#     def get_results(self):
#         return self.params



def create_data_recorder(params: Params):
    components = [ParamRecorder(params)]
    visited_segments = None
    if params.is_recording_position:
        components.append(PositionRecorder(params))
    if params.is_recording_area:
        rec = AreaGridRecorder(params)
        components.append(rec)
        visited_segments = rec.visited_spaces
    if params.is_recording_area_indices:
        components.append(AreaIndexRecorder(params))
    if params.is_recording_area_over_time:
        rec = AreaOverTimeRecorder(params)
        components.append(rec)
        visited_segments = rec.visited_spaces
    if params.is_recording_loss:
        components.append(LossRecorder(params))
#    if params.is_saving_network:
#        components.append(NetworkSaver(params))

    return DataRecorder(components), visited_segments
