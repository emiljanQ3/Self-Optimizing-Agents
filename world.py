import numpy as np
from tags import WorldTag
import utils


def create_world(params):
    components = []
    if len(params.viscosity_times) > 0:
        components.append(ChangingViscosity(params))
    if params.selected_world == WorldTag.EMPTY:
        components.extend([])
    if params.selected_world == WorldTag.EMPTY_REPEATING:
        components.extend([RepeatingBoundaryConditions()])
    if params.selected_world == WorldTag.CONVEX_CELLS:
        components.extend([ConvexCells(params)])
    if params.selected_world == WorldTag.CONCAVE_CELLS:
        components.extend([ConcaveCells(params)])
    if params.selected_world == WorldTag.CIRCLE:
        components.extend([SingleCircle(params)])

    return World(components)


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


class ChangingViscosity:

    def __init__(self, params):
        self.i = 0
        self.timer = 0
        self.base_speed = params.speed
        self.base_delta_t = params.delta_time

    def apply(self, agents, new_agents, params):
        if self.timer == 0:
            speed_factor, timer = params.viscosity_times[self.i]
            params.speed = self.base_speed*speed_factor
            params.delta_time = self.base_delta_t/speed_factor

            self.timer = timer
            self.i = (self.i + 1) % len(params.viscosity_times)

        return agents, new_agents


class ConvexCells:
    def __init__(self, params):
        self.r = params.obstacle_size*params.cell_size/2

    def apply(self, agents, new_agents, params):
        center_cell_pos = np.mod(new_agents[:, :2], params.cell_size) - np.ones([1, 2]) * params.cell_size / 2
        rho, phi = utils.cart2pol(center_cell_pos[:, 0], center_cell_pos[:, 1])
        rho = np.maximum(rho, self.r)
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


class SingleCircle:
    def __init__(self, params):
        self.r = params.cell_size

    def apply(self, agents, new_agents, params):
        center_cell_pos = new_agents[:, :2]
        rho, phi = utils.cart2pol(center_cell_pos[:, 0], center_cell_pos[:, 1])
        rho = np.minimum(rho, self.r)
        x, y = utils.pol2cart(rho, phi)

        new_agents[:, 0] = x
        new_agents[:, 1] = y

        return agents, new_agents

