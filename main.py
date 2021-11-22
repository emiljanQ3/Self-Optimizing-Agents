from world import create_world
from move import create_mover
from simulation import simulate
from record import create_data_recorder
from disk import save, load_all
from plot import display_results
from config import Params
import numpy as np
from plot import plot_area_over_alpha


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


if __name__ == '__main__':
    params = Params()
    # run_param_search(params)
    plot_area_over_alpha(load_all(params))
