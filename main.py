from world import create_world
from move import create_mover
from simulation import simulate
from record import create_data_recorder
from disk import save, load_all
import plot
from config import Params
import numpy as np
import pickle
from plot import plot_area_over_alpha
from tags import ResultTag, AlphaInitTag, MoveTag
import copy
from multiprocessing import Pool
from utils import normalize_area_to_best_alpha
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map
from data import DataModifier
from genetic_keras.plot.plot import epoch_hist_plot


def run_simulation(params):
    world = create_world(params)
    mover = create_mover(params)
    data_recorder, visited_segments = create_data_recorder(params)
    data_modifier = DataModifier(visited_segments, params)
    results = simulate(world, mover, data_recorder, data_modifier, params)
    save(results, params)
    #display_results(results, params)


def run_contrast_param_search(params: Params):

    params_list = []

    slow_optimal_params = copy.deepcopy(params)
    slow_optimal_params.save_id += "_slow_optimal"
    slow_optimal_params.selected_mover = MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST
    slow_optimal_params.alpha = 2.2
    params_list.append(slow_optimal_params)

    instant_optimal_params = copy.deepcopy(params)
    instant_optimal_params.save_id += "_instant_optimal"
    instant_optimal_params.selected_mover = MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST_INSTANT_SWITCH
    instant_optimal_params.alpha = 2.3
    params_list.append(instant_optimal_params)

    for alpha in np.linspace(1, 2, 11):
        temp_params = copy.deepcopy(params)
        temp_params.alpha = alpha

        temp_params.save_id += f"_a{alpha:.1f}"
        temp_params.selected_mover = MoveTag.LEVY_VARYING_DELTA_CONTRAST
        params_list.append(temp_params)

    process_map(run_simulation, params_list)
    # it = [run_simulation(x) for x in params_list]


def run_alpha_linspace(params:Params):
    params_list = []

    for alpha in np.linspace(1, 2, 11):
        temp_params = copy.deepcopy(params)
        temp_params.alpha = alpha

        temp_params.save_id += f"{params.selected_world}_a{alpha:.1f}-"
        temp_params.selected_mover = MoveTag.LEVY_VARYING_DELTA_CONTRAST
        params_list.append(temp_params)

    process_map(run_simulation, params_list)
    # it = [run_simulation(x) for x in params_list]


def run_paralell(params: Params, num_simulations):

    params_list = []

    for i in range(num_simulations):
        temp_params = copy.deepcopy(params)

        temp_params.save_id += f"-1"
        params_list.append(temp_params)

    process_map(run_simulation, params_list)


def run_genetic_validations(params: Params):
    params_list = []

    r0 = params.tic_rate_0
    r1 = params.tic_rate_1

    for i in range(5):
        temp_params = copy.deepcopy(params)

        temp_params.save_id += f"{i}-"
        temp_params.model_location = \
            f"thesis_data/genetic_training/genetic_run_{r0}_{r1}/genetic_run_{r0}_{r1}{i}/genetic_run_{r0}_{r1}{i}"

        params_list.append(temp_params)

    process_map(run_simulation, params_list)


if __name__ == '__main__':
    params = Params()
    run_alpha_linspace(params)
