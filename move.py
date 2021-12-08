import math

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
    return np.power(np.random.random_sample(size), -1 * (3 - params.alpha))


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


class AlphaChanger:

    def __init__(self, params):
        self.i = 0
        self.timer = 0

    def apply(self, agents, new_agents, params):
        if self.timer == 0:
            params.alpha, self.timer = params.alpha_times[self.i]

            self.i = (self.i + 1) % len(params.viscosity_times)

        self.timer -= 1

        return agents, new_agents


def delta_variation(x, width):
    return 2 ** cell_exponent(x, width)


def big_contrast_delta_variation(x):
    cells = np.mod(np.floor(x), 2)
    cells[cells == 0] = 3
    cells[cells == 1] = -4
    return 2 ** cells


def cell_exponent(x, width):
    return (np.floor(
                abs(
                    np.mod(
                        x, width
                    ) - (width / 2)
                )
            ) - (width / 2 - 1) / 2)


class LevyRotaterVaryingDelta:
    def __init__(self, levy_dist_sampler, params):
        self.levy_timer = levy_dist_sampler(params.num_agents, params)
        self.dist_sampler = levy_dist_sampler

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * delta_variation(x=agents[:, 0], width=params.world_width)

        is_turning = self.levy_timer <= 0
        self.levy_timer[is_turning] = self.dist_sampler(is_turning.sum(), params)
        agents[:, 2][is_turning] += np.random.standard_normal(is_turning.sum()) * params.ang_sd

        return agents, agents_data


class AlwaysOptimalLevyRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)
        self.optimal_alphas = {6: 1.0, 5: 1.0, 4: 1.0, 3: 1.0, 2: 1.1, 1: 1.2, 0: 1.2,
                               -1: 1.5, -2: 2.0, -3: 1.9, -4: 2.0, -5: 2.0, -6: 2.0}

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * delta_variation(x=agents[:, 0], width=params.world_width)
        turning_idx = np.argwhere(self.levy_timer <= 0)
        for i in turning_idx:
            self.levy_timer[i] = np.abs(levy_stable.rvs(
                alpha=self.optimal_alphas[cell_exponent(agents[i, 0].item(), params.world_width)], beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data


class BigContrastLevyRotaterVaryingDelta:
    def __init__(self, levy_dist_sampler, params):
        self.levy_timer = levy_dist_sampler(params.num_agents, params)
        self.dist_sampler = levy_dist_sampler

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0])

        is_turning = self.levy_timer <= 0
        self.levy_timer[is_turning] = self.dist_sampler(is_turning.sum(), params)
        agents[:, 2][is_turning] += np.random.standard_normal(is_turning.sum()) * params.ang_sd

        return agents, agents_data


class BigContrastOptimalLevyRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)
        self.optimal_alphas = {3: 1.0, -4: 2.0}

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0])
        turning_idx = np.argwhere(self.levy_timer <= 0)
        for i in turning_idx:
            self.levy_timer[i] = np.abs(levy_stable.rvs(
                alpha=math.floor(agents[i, 0].item()) % 2 + 1, beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data


class AgentSpecificLevyRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time
        turning_idx = np.argwhere(self.levy_timer <= 0)
        for i in turning_idx:
            self.levy_timer[i] = np.abs(levy_stable.rvs(alpha=agents_data.alphas[i], beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data


class ExactLevyMover:
    def __init__(self, params):
        self.levy_timer = sample_levy(params.num_agents, params)

    def apply(self, agents, agents_data, params):
        remaining_delta_time = params.delta_time * np.ones(params.num_agents)
        delta_pos = np.zeros((params.num_agents, 3))
        is_turning = self.levy_timer < remaining_delta_time
        while any(is_turning):
            remaining_delta_time[is_turning] -= self.levy_timer[is_turning]
            delta_pos[:, 0][is_turning] += np.cos(delta_pos[:, 2][is_turning]) * params.speed * self.levy_timer[
                is_turning]
            delta_pos[:, 1][is_turning] += np.sin(delta_pos[:, 2][is_turning]) * params.speed * self.levy_timer[
                is_turning]
            self.levy_timer[is_turning] = sample_levy(is_turning.sum(), params)
            delta_pos[:, 2][is_turning] += np.random.standard_normal(is_turning.sum()) * params.ang_sd
            is_turning = self.levy_timer < remaining_delta_time

        delta_pos[:, 0] += np.cos(delta_pos[:, 2]) * params.speed * remaining_delta_time
        delta_pos[:, 1] += np.sin(delta_pos[:, 2]) * params.speed * remaining_delta_time

        return agents + delta_pos, agents_data


def create_mover(params):
    components = []
    if len(params.alpha_times) > 0:
        components.append(AlphaChanger(params))
    if params.selected_mover == MoveTag.BROWNIAN:
        components.extend([BrownianTranslation()])
    if params.selected_mover == MoveTag.ACTIVE_ROTDIFF:
        components.extend([BrownianRotation(), ForwardMovement()])
    if params.selected_mover == MoveTag.AYKUT_LEVY:
        components.extend([LevyRotater(sample_levy_aykut, params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY:
        if params.alpha_tag == AlphaInitTag.SAME:
            components.extend([LevyRotater(sample_levy, params), ForwardMovement()])
        else:
            components.extend([AgentSpecificLevyRotater(params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY_VARYING_DELTA:
        components.extend([LevyRotaterVaryingDelta(sample_levy, params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY_OPTIMAL_ALPHA:
        components.extend([AlwaysOptimalLevyRotater(params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY_VARYING_DELTA_CONTRAST:
        components.extend([BigContrastLevyRotaterVaryingDelta(sample_levy, params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST:
        components.extend([BigContrastOptimalLevyRotater(params), ForwardMovement()])

    return Mover(components)
