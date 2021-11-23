from world import create_world
from move import create_mover
from simulation import simulate
from record import create_data_recorder
from disk import save, load_all
from plot import display_results
from config import Params
import numpy as np
from plot import plot_area_over_alpha
from tags import ResultTag, AlphaInitTag
import copy


def run_simulation(params):
    world = create_world(params)
    mover = create_mover(params)
    data_recorder = create_data_recorder(params)
    results = simulate(world, mover, data_recorder, params)
    save(results, params)
    display_results(results, params)


def run_param_search(params):

    for alpha in np.linspace(1, 2, 11):
        params.alpha = alpha
        run_simulation(params)


def rerun_saved_run(results):
    rerun_params = copy.deepcopy(results[0][ResultTag.PARAM])
    rerun_params.num_agents = 5
    rerun_params.num_repeats = 1
    rerun_params.alpha_tag = AlphaInitTag.LINSPACE
    rerun_params.is_recording_position = True
    rerun_params.is_recording_area = False
    rerun_params.is_recording_area_indices = True
    rerun_params.is_plotting_trajectories = True
    rerun_params.is_plotting_area_units = True
    rerun_params.save_id += "_rerun"
    run_simulation(rerun_params)

    plot_area_over_alpha(results)


if __name__ == '__main__':
    params = Params()
    run_param_search(params)
    plot_area_over_alpha(load_all(params))
