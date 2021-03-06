import numpy as np
from tags import AlphaInitTag
from data import AgentsData
from neural_networks import GenNetContainer, NeuralNetworkContainer
from tensorflow import keras
from genetic_keras import utils
import pickle


def init_agents_pos(params):
    random = np.random.random_sample([params.num_agents, 3])
    random[:, 0] *= params.world_width
    random[:, 1] *= params.world_height
    random[:, 2] *= 2*np.pi
    return random


def init_agents_data(params, models):
    if params.alpha_tag == AlphaInitTag.SAME:
        return AgentsData(alphas=np.ones(params.num_agents) * params.alpha, params=params)
    if params.alpha_tag == AlphaInitTag.LINSPACE:
        return AgentsData(alphas=np.linspace(1, 2, params.num_agents), params=params)
    if params.alpha_tag == AlphaInitTag.NETWORK and models is not None:
        return AgentsData(alphas=None, params=params, network_containers=[GenNetContainer(params, m) for m in models])
    if params.alpha_tag == AlphaInitTag.NETWORK and params.is_backprop_training:
        return AgentsData(alphas=None, params=params,
                          network_containers=[NeuralNetworkContainer(params) for _ in range(params.num_agents)])
    if params.alpha_tag == AlphaInitTag.NETWORK and not params.is_backprop_training:
        model: keras.Model = keras.models.load_model(params.model_location + "_sg.mdl")
        weights = model.get_weights()
        with open(params.model_location + '.pkl', 'rb') as file:
            history = pickle.load(file)[0]
        chromosome = history[-1].best_chromosome_training
        utils.set_params(model, chromosome, weights)
        container = GenNetContainer(params, model)
        return AgentsData(alphas=None, params=params, network_containers=[container for _ in range(params.num_agents)])

