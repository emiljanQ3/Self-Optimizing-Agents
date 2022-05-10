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


def plot_local_optima():
    params = Params()
    params.results_path = "thesis_data/2D_plot/all_combined/"
    params.save_id = ""
    results, _ = load_all(params)
    plot.plot_alpha_delta_surface(results)


def plot_overview(folder):
    params = Params()
    params.results_path = f"thesis_data/overviews/{folder}/"
    params.save_id = ""
    results, file_names = load_all(params)
    plot.plot_area_in_range(results, 0, 100000-1, file_names, f"{folder}")


def plot_compressed_overview(r0, r1, force_recalculation=False):
    params = Params()
    params.results_path = f"thesis_data/overviews/overview_{r0}_{r1}/"
    params.save_id = ""
    params.tic_rate_0 = r0
    params.tic_rate_1 = r1
    plot.plot_top_contenders(params, force_recalculation)


def plot_genetic_training_history(r0, r1):
    params = Params()
    params.results_path = f"thesis_data/genetic_training/genetic_run_{r0}_{r1}/"
    params.save_id = f"genetic_run_{r0}_{r1}"
    params.tic_rate_0 = r0
    params.tic_rate_1 = r1
    plot.plot_genetic_training_history(params)


def plot_validation_distribution(r0, r1):
    params = Params()
    params.results_path = f"thesis_data/overviews/overview_{r0}_{r1}/"
    params.save_id = "gen"
    params.tic_rate_0 = r0
    params.tic_rate_1 = r1

    plot.plot_distribution(params)


if __name__ == '__main__':
    #plot_local_optima()
    #plot_overview("overview_-4_-6")
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
    for e in environments:
        #plot_compressed_overview(*e)
        #plot_genetic_training_history(*e)
        plot_validation_distribution(*e)



    plt.show()