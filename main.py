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


def run_param_search(params: Params):

    params_list = []
    for v in np.logspace(-6, 6, num=13, base=2):
        for alpha in np.linspace(1, 2, 11):
            temp_params = copy.deepcopy(params)
            temp_params.alpha = alpha
            temp_params.speed *= v

            step_size = temp_params.area_unit_size/2
            time = temp_params.delta_time * temp_params.num_steps

            temp_params.delta_time = step_size/temp_params.speed
            temp_params.num_steps = time/temp_params.delta_time

            temp_params.save_id += f"_v{v}_a{alpha}"
            params_list.append(temp_params)


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
