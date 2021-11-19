from world import create_world
from move import create_mover
from simulation import simulate
from record import create_data_recorder
from disk import save
from plot import display_results
from config import Params


def run_simulation(params):
    world = create_world(params)
    mover = create_mover(params)
    data_recorder = create_data_recorder(params)
    results = simulate(world, mover, data_recorder, params)
    save(results, params)
    display_results(results, params)

if __name__ == '__main__':
    params = Params()
    run_simulation(params)

