import numpy as np
from tensorflow import keras
from genetic_keras.utils import set_params
from genetic_keras.nbn_training import set_nbn_params
from multiprocessing import Process, cpu_count, JoinableQueue
from world import create_world
from move import create_mover
from simulation import simulate
from record import create_data_recorder
from disk import save, load_all
import plot
from config import Params
from plot import plot_area_over_alpha
from tags import ResultTag, AlphaInitTag, MoveTag
import copy
from multiprocessing import Pool
from utils import normalize_area_to_best_alpha
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map
from data import DataModifier
from config import Params

params = Params()


def run_simulation(models):
    world = create_world(params)
    mover = create_mover(params)
    data_recorder, visited_segments = create_data_recorder(params)
    data_modifier = DataModifier(visited_segments, params)
    results = simulate(world, mover, data_recorder, data_modifier, params, models)
    return results[ResultTag.AREA]


def evaluate_population(population:np.ndarray, model: keras.Model):
    models = [keras.models.clone_model(model) for _ in range(len(population))]

    for i in range(len(population)):
        weights = models[i].get_weights()
        set_params(models[i], population[i], weights)

    scores = run_simulation(models)
    scores = np.mean(scores, axis=(1, 2))
    return -scores


def test_map(f, xs):
    return [f(x) for x in xs]




