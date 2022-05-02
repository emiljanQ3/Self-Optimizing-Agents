import math

import numpy as np
from scipy.stats import levy_stable
from tags import MoveTag, AlphaInitTag
from config import Params

optimal_alphas = {-6.0: 2.0, -5.5: 2.0, -5.0: 2.0, -4.5: 2.0, -4.0: 1.85, -3.5: 1.75, -3.0: 1.5, -2.5: 1.3,
                  -2.0: 1.35, -1.5: 1.35, -1.0: 1.2, -0.5: 1.2, 0.0: 1.1, 0.5: 1.15, 1.0: 1.1, 1.5: 1.05,
                  2.0: 1.0, 2.5: 1.0, 3.0: 1.0, 3.5: 1.0, 4.0: 1.0, 4.5: 1.0, 5.0: 1.0, 5.5: 1.0, 6.0: 1.0}


class Mover:
    def __init__(self, components):
        self.components = components

    def step(self, agents, agents_data, params):
        for c in self.components:
            agents, agents_data = c.apply(agents, agents_data, params)

        return agents, agents_data


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


def delta_variation(x, width):
    return 2 ** cell_exponent(x, width)


def big_contrast_delta_variation(x, params: Params):
    cells = np.mod(np.floor(x), 2)
    cells[cells == 0] = params.tic_rate_0
    cells[cells == 1] = params.tic_rate_1
    return 2 ** cells


def cell_exponent(x, width):
    return (np.floor(
                abs(
                    np.mod(
                        x, width
                    ) - (width / 2)
                )
            ) - (width / 2) / 2)


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

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * delta_variation(x=agents[:, 0], width=params.world_width)
        turning_idx = np.argwhere(self.levy_timer <= 0)
        for i in turning_idx:
            self.levy_timer[i] = np.abs(levy_stable.rvs(
                alpha=optimal_alphas[cell_exponent(agents[i, 0].item(), params.world_width)], beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data


class BigContrastLevyRotaterVaryingDelta:
    def __init__(self, levy_dist_sampler, params):
        self.levy_timer = levy_dist_sampler(params.num_agents, params)
        self.dist_sampler = levy_dist_sampler

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0], params=params)

        is_turning = self.levy_timer <= 0
        self.levy_timer[is_turning] = self.dist_sampler(is_turning.sum(), params)
        agents[:, 2][is_turning] += np.random.standard_normal(is_turning.sum()) * params.ang_sd

        return agents, agents_data


class BigContrastOptimalLevyRotater:
    def __init__(self, params, instant_switch=False):
        self.levy_timer = np.zeros(params.num_agents)
        self.instant_switch = instant_switch
        self.last_x = 0

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0], params=params)
        turning_idx = np.argwhere(np.logical_or(self.levy_timer <= 0,  self.apply_instant_switch(agents)))
        for i in turning_idx:
            self.levy_timer[i] = np.abs(levy_stable.rvs(
                alpha=optimal_alphas[params.tic_rate_0] if math.floor(agents[i, 0].item()) % 2 == 0 else optimal_alphas[params.tic_rate_1], beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data

    def apply_instant_switch(self, agents):
        if not self.instant_switch:
            return 0
        this_x = np.floor(agents[:, 0])
        answer = this_x != self.last_x
        self.last_x = this_x
        return answer


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


class BigContrastNeuralNetworkLevyRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)
        self.is_resetting_reward = params.is_backprop_training

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0], params=params)
        turning_idx = np.nonzero(self.levy_timer <= 0)[0]
        for i in turning_idx:
            mean_reward = agents_data.reward_since_last_action[i]/agents_data.steps_since_last_action[i]
            agents_data.network_containers[i].train_network(1)
            alpha = agents_data.network_containers[i].get_next_alpha(agents_data.memory[i], mean_reward)
            self.levy_timer[i] = np.abs(levy_stable.rvs(alpha=alpha, beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

            if self.is_resetting_reward:
                agents_data.reward_since_last_action[i] = 0
                agents_data.steps_since_last_action[i] = 0

        return agents, agents_data


class BigContrastNeuralNetworkDirectTimerRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)
        self.is_resetting_reward = params.is_backprop_training
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)

    def notify_listeners(self, timer, agent):
        for l in self.listeners:
            l.receive(timer, agent)

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0], params=params)
        turning_idx = np.nonzero(self.levy_timer <= 0)[0]
        for i in turning_idx:
            mean_reward = agents_data.reward_since_last_action[i]/agents_data.steps_since_last_action[i]
            agents_data.network_containers[i].train_network(1)
            self.levy_timer[i] = agents_data.network_containers[i].get_next_alpha(agents_data.memory[i], mean_reward)
            self.notify_listeners(self.levy_timer[i], agents[i])
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

            if self.is_resetting_reward:
                agents_data.reward_since_last_action[i] = 0
                agents_data.steps_since_last_action[i] = 0

        return agents, agents_data


class PretrainedNetworkLevyRotater:
    def __init__(self, params):
        self.levy_timer = np.zeros(params.num_agents)

    def apply(self, agents, agents_data, params):
        self.levy_timer -= params.delta_time * big_contrast_delta_variation(x=agents[:, 0], params=params)
        turning_idx = np.nonzero(self.levy_timer <= 0)[0]
        for i in turning_idx:
            alpha = agents_data.network_containers[i].get_next_alpha(agents_data.memory[i], None, extend_buffer=False)
            self.levy_timer[i] = np.abs(levy_stable.rvs(alpha=alpha, beta=0))
            agents[i, 2] += np.random.standard_normal() * params.ang_sd

        return agents, agents_data


def create_mover(params):
    components = []

    if params.selected_mover == MoveTag.ACTIVE_ROTDIFF:
        components.extend([BrownianRotation(), ForwardMovement()])
    if params.selected_mover == MoveTag.AYKUT_LEVY:
        components.extend([LevyRotater(sample_levy_aykut, params), ForwardMovement()])
    if params.selected_mover == MoveTag.LEVY:
        if params.alpha_tag == AlphaInitTag.SAME:
            rotator = LevyRotater(sample_levy, params)
            components.extend([rotator, ForwardMovement()])
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
    if params.selected_mover == MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST_INSTANT_SWITCH:
        components.extend([BigContrastOptimalLevyRotater(params, instant_switch=True), ForwardMovement()])
    if params.selected_mover == MoveTag.NEURAL_LEVY and params.is_backprop_training:
        components.extend([BigContrastNeuralNetworkLevyRotater(params), ForwardMovement()])
    if params.selected_mover == MoveTag.NEURAL_LEVY and not params.is_backprop_training:
        components.extend([PretrainedNetworkLevyRotater(params), ForwardMovement()])
    if params.selected_mover == MoveTag.DIRECT_TIMER:
        components.extend([BigContrastNeuralNetworkDirectTimerRotater(params), ForwardMovement()])

    return Mover(components)
