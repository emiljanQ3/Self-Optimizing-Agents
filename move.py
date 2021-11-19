import numpy as np
from scipy.stats import levy_stable
from tags import MoveTag, AlphaInitTag

class Mover:
    def __init__(self, components):
        self.components = components

    def step(self, agents, agents_data, params):
        for c in self.components:
            agents, agents_data = c.apply(agents, agents_data, params)

        return agents, agents_data


class BrownianTranslation:
    def apply(self, agents, agents_data, params):
        return add_noise(agents, [params.trans_sd, params.trans_sd, 0]), agents_data


class BrownianRotation:
    def apply(self, agents, agents_data, params):
        return add_noise(agents, [0, 0, params.ang_sd]), agents_data


def add_noise(agents, levels):
    standard_noise = np.random.standard_normal(np.shape(agents))
    noise_levels = np.array(levels)
    new_agents = agents + standard_noise * noise_levels
    return new_agents


class ForwardMovement:
    def apply(self, agents, agents_data, params):
        agents[:, 0] = agents[:, 0] + np.cos(agents[:, 2]) * params.speed * params.delta_time
        agents[:, 1] = agents[:, 1] + np.sin(agents[:, 2]) * params.speed * params.delta_time
        return agents, agents_data


def sample_levy_aykut(size, params):
    return np.power(np.random.random_sample(size),-1*(3-params.alpha))


def sample_levy(size, params):
    return np.abs(levy_stable.rvs(params.alpha, beta=0, size=size))


class LevyRotater:
    def __init__(self, levy_dist_sampler, params):
        self.levy_timer = levy_dist_sampler(params.num_agents, params)
        self.dist_sampler = levy_dist_sampler

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time
        is_turning = self.levy_timer <= 0
        self.levy_timer[is_turning] = self.dist_sampler(is_turning.sum(), params)
        agents[:, 2][is_turning] += np.random.standard_normal(is_turning.sum()) * params.ang_sd

        return agents, agents_data


class AgentSpecificLevyRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time
        turning_idx = np.argwhere(self.levy_timer <= 0)
        for i in turning_idx:
            self.levy_timer[i] = levy_stable.rvs(alpha=agents_data.alphas[i], beta=0)
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data


def create_mover(params):
    if params.selected_mover == MoveTag.BROWNIAN:
        return Mover([BrownianTranslation()])
    if params.selected_mover == MoveTag.ACTIVE_ROTDIFF:
        return Mover([BrownianRotation(), ForwardMovement()])
    if params.selected_mover == MoveTag.AYKUT_LEVY:
        return Mover([LevyRotater(sample_levy_aykut, params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY:
        if params.alpha_tag == AlphaInitTag.SAME:
            return Mover([LevyRotater(sample_levy, params), ForwardMovement()])
        else:
            return Mover([AgentSpecificLevyRotater(params), ForwardMovement()])

    raise Exception("Invalid movetag in parameters.")
