import numpy as np
from tags import WorldTag
import utils


def create_world(params):
    if params.selected_world == WorldTag.EMPTY:
        return World([])
    if params.selected_world == WorldTag.EMPTY_REPEATING:
        return World([RepeatingBoundaryConditions()])
    if params.selected_world == WorldTag.CONVEX_CELLS:
        return World([ConvexCells(params)])
    if params.selected_world == WorldTag.CONCAVE_CELLS:
        return World([ConcaveCells(params)])

    raise Exception("Invalid worldtag in parameters.")

class World:
    def __init__(self, components):
        self.components = components

    def step(self, agents, new_agents, params):
        for c in self.components:
            agents, new_agents = c.apply(agents, new_agents, params)

        return new_agents


class RepeatingBoundaryConditions:
    def apply(self, agents, new_agents, params):
        new_agents[:, 0] = np.mod(new_agents[:, 0], params.world_width)
        new_agents[:, 1] = np.mod(new_agents[:, 1], params.world_height)
        return agents, np.mod(new_agents)

class ConvexCells:
    def __init__(self, params):
        self.r = params.obstacle_size*params.cell_size/2

    def apply(self, agents, new_agents, params):
        center_cell_pos = np.mod(new_agents[:, :2], params.cell_size) - np.ones([1, 2]) * params.cell_size / 2
        rho, phi = utils.cart2pol(center_cell_pos[:, 0], center_cell_pos[:, 1])
        rho = np.minimum(rho, self.r)
        x, y = utils.pol2cart(rho, phi)
        diff = - center_cell_pos
        diff[:, 0] += x
        diff[:, 1] += y

        new_agents[:, :2] += diff

        return agents, new_agents

class ConcaveCells:
    def __init__(self, params):
        x = params.obstacle_size
        h = np.sqrt((np.power(1-x, 2) + 1)/4)
        self.r = h * params.cell_size

    def apply(self, agents, new_agents, params):
        center_cell_pos = np.mod(new_agents[:, :2], params.cell_size) - np.ones([1, 2]) * params.cell_size / 2
        rho, phi = utils.cart2pol(center_cell_pos[:, 0], center_cell_pos[:, 1])
        rho = np.minimum(rho, self.r)
        x, y = utils.pol2cart(rho, phi)
        diff = - center_cell_pos
        diff[:, 0] += x
        diff[:, 1] += y

        new_agents[:, :2] += diff

        return agents, new_agents

