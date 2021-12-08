from world import create_world
from move import create_mover
from simulation import simulate
from record import create_data_recorder
from disk import save, load_all
from plot import display_results, plot_alpha_speed_surface, scatter_alpha_speed_surface, plot_many_area_over_time
from config import Params
import numpy as np
from plot import plot_area_over_alpha
from tags import ResultTag, AlphaInitTag
import copy
from multiprocessing import Pool
from utils import normalize_area_to_best_alpha
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map


def run_simulation(params):
    world = create_world(params)
    mover = create_mover(params)
    data_recorder = create_data_recorder(params)
    results = simulate(world, mover, data_recorder, params)
    save(results, params)
    display_results(results, params)


def run_param_search(params: Params):

    params_list = []
    for v in [1]:
        for alpha in np.linspace(1, 2, 11):
            temp_params = copy.deepcopy(params)
            temp_params.alpha = alpha
            temp_params.speed *= v

            step_size = temp_params.area_unit_size / 10

            temp_params.delta_time = step_size / temp_params.speed

            temp_params.alpha_times = []

            temp_params.save_id += f"_v{v}_a{alpha}"
            params_list.append(temp_params)

    temp_params = copy.deepcopy(params)
    temp_params.save_id += "_x"
    params_list.append(temp_params)

    process_map(run_simulation, params_list)
    #it = [run_simulation(x) for x in params_list]


if __name__ == '__main__':
    params = Params()
    run_param_search(params)
    plot_many_area_over_time(load_all(params))
    #plot_area_over_alpha(load_all(params))
    #plot_alpha_speed_surface(load_all(params))
    #plot_alpha_speed_surface(normalize_area_to_best_alpha(load_all(params)))
    #scatter_alpha_speed_surface(load_all(params))
    #run_simulation(params)
    #plt.show()
