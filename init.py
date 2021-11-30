import numpy as np
from tags import AlphaInitTag
from data import AgentsData


def init_agents_pos(params):
    random = np.random.random_sample([params.num_agents, 3])
    random[:, 0] *= params.world_width
    random[:, 1] *= params.world_height
    random[:, 2] *= 2*np.pi
    return random


def init_agents_data(params):
    if params.alpha_tag == AlphaInitTag.SAME:
        return AgentsData(alphas=np.ones(params.num_agents) * params.alpha)
    if params.alpha_tag == AlphaInitTag.LINSPACE:
        return AgentsData(alphas=np.linspace(1, 2, params.num_agents))

