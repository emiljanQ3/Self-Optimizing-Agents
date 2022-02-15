import numpy as np
from tags import AlphaInitTag
from data import AgentsData
from neural_networks import GenNetContainer, NeuralNetworkContainer
from tensorflow import keras


def init_agents_pos(params):
    random = np.random.random_sample([params.num_agents, 3])
    random[:, 0] *= params.world_width
    random[:, 1] *= params.world_height
    random[:, 2] *= 2*np.pi
    return random


def init_agents_data(params, model):
    if params.alpha_tag == AlphaInitTag.SAME:
        return AgentsData(alphas=np.ones(params.num_agents) * params.alpha, params=params)
    if params.alpha_tag == AlphaInitTag.LINSPACE:
        return AgentsData(alphas=np.linspace(1, 2, params.num_agents), params=params)
    if params.alpha_tag == AlphaInitTag.NETWORK and model is not None:
        container = GenNetContainer(params)
        container.network = model
        return AgentsData(alphas=None, params=params, network_containers=[container for _ in range(params.num_agents)])
    if params.alpha_tag == AlphaInitTag.NETWORK and params.is_backprop_training:
        return AgentsData(alphas=None, params=params,
                          network_containers=[NeuralNetworkContainer(params) for _ in range(params.num_agents)])
    if params.alpha_tag == AlphaInitTag.NETWORK and not params.is_backprop_training:
        model = keras.models.load_model("model1")
        container = NeuralNetworkContainer(params)
        container.network = model
        return AgentsData(alphas=None, params=params, network_containers=[container for _ in range(params.num_agents)])

