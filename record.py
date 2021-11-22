import numpy as np
from tags import ResultTag


class DataRecorder:
    def __init__(self, components):
        self.components = components

    def record(self, agents, new_agents, world, mover, rep, step):
        for c in self.components:
            c.record(agents, new_agents, world, mover, rep, step)

    def get_results(self):
        results = dict()
        for c in self.components:
            results[c.tag] = c.get_results()

        return results


class PositionRecorder:
    def __init__(self, params):
        self.positions_over_time = -np.ones([params.num_repeats, params.num_steps, params.num_agents, 2])
        self.tag = ResultTag.POSITION

    def record(self, agents, new_agents, world, mover, rep, step):
        self.positions_over_time[rep, step] = agents[:, :2]

    def get_results(self):
        return self.positions_over_time


class AreaGridRecorder:
    def __init__(self, params):
        self.tag = ResultTag.AREA
        self.visited_spaces = [set() for i in range(params.num_agents)]
        self.area_unit_size = params.area_unit_size

    def record(self, agents, new_agents, world, mover, rep, step):
        for i in range(np.shape(agents)[0]):
            area_unit_pos = np.floor(agents[i, :2] / self.area_unit_size)
            int_pos = (int(area_unit_pos[0]), int(area_unit_pos[1]))
            self.visited_spaces[i].add(int_pos)

    def get_results(self):
        area = [len(x)*self.area_unit_size**2 for x in self.visited_spaces]
        return area


class ParamRecorder:
    def __init__(self, params):
        self.tag = ResultTag.PARAM
        self.params = params

    def record(self, agents, new_agents, world, mover, rep, step):
        pass

    def get_results(self):
        return self.params


def create_data_recorder(params):
    components = [ParamRecorder(params)]
    if params.is_recording_position:
        components.append(PositionRecorder(params))
    if params.is_recording_area:
        components.append(AreaGridRecorder(params))

    return DataRecorder(components)
