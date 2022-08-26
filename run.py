import sys

import tags
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

environments = [
                (-4, -6),
                (-2, -6),
                (0, -6),
                (2, -6),
                (3, -4),
                (4, -3),
                (6, -6),
                (6, -2),
                (6, 0),
                (6, 2),
                (6, 4)
                ]

def run_simulation(params):
    world = create_world(params)
    mover = create_mover(params)
    data_recorder, visited_segments = create_data_recorder(params)
    data_modifier = DataModifier(visited_segments, params)
    results = simulate(world, mover, data_recorder, data_modifier, params)
    save(results, params)
    #display_results(results, params)


def run_overview(params: Params, skip_opti=False):

    params_list = []

    if not skip_opti:
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
    if sys.argv[1] == "0":
        params.selected_world = tags.WorldTag.CONCAVE_CELLS
    elif sys.argv[1] == "1":
        params.selected_world = tags.WorldTag.CONVEX_CELLS
    elif sys.argv[1] == "2":
        params.selected_world = tags.WorldTag.EMPTY
    else:
        print("invalid job argument")
        return

    params_list = []

    for alpha in np.linspace(1, 2, 11):
        temp_params = copy.deepcopy(params)
        temp_params.alpha = alpha

        temp_params.save_id += f"{params.selected_world}_a{alpha:.1f}-"
        temp_params.selected_mover = MoveTag.LEVY_VARYING_DELTA_CONTRAST
        params_list.append(temp_params)

    process_map(run_simulation, params_list)
    # it = [run_simulation(x) for x in params_list]


def run_alpha_r_surface(params:Params):
    params_list = []

    for alpha in np.linspace(1, 2, 21):
        for r in np.linspace(-6, 6, 25):
            temp_params = copy.deepcopy(params)
            temp_params.alpha = alpha
            temp_params.tic_rate_0 = r
            temp_params.tic_rate_1 = r

            temp_params.save_id += f"2d_a{alpha:.1f}_r{r}-"
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


def opti_reruns(params: Params):
    params_list = []
    for env in environments:
        temp_opti = copy.deepcopy(params)
        temp_opti_s = copy.deepcopy(params)

        temp_opti.selected_mover = MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST
        temp_opti_s.selected_mover = MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST_INSTANT_SWITCH

        temp_opti.tic_rate_0 = env[0]
        temp_opti.tic_rate_1 = env[1]
        temp_opti.save_id += "opti_" + str(env)

        params_list.append(temp_opti)

        temp_opti_s.tic_rate_0 = env[0]
        temp_opti_s.tic_rate_1 = env[1]
        temp_opti_s.save_id += "opti_s_" + str(env)

        params_list.append(temp_opti_s)

    process_map(run_simulation, params_list)


if __name__ == '__main__':
    params = Params()
    opti_reruns(params)
