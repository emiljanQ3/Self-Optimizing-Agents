import numpy as np
from tags import AlphaInitTag
from data import AgentsData


def init_agents_pos(params):
    return np.random.random_sample([params.num_agents, 3])


def init_agents_data(params):
    if params.alpha_tag == AlphaInitTag.SAME:
        return AgentsData(alphas=np.ones(params.num_agents) * params.alpha)
    if params.alpha_tag == AlphaInitTag.LINSPACE:
        return AgentsData(alphas=np.linspace(1, 2, params.num_agents))

