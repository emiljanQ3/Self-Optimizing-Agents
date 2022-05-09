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


def plot_compressed_overview(folder, force_recalculation=False):
    params = Params()
    params.results_path = f"thesis_data/overviews/{folder}/"
    params.save_id = ""
    plot.plot_top_contenders(params, force_recalculation)

if __name__ == '__main__':
    #plot_local_optima()
    #plot_overview("overview_-4_-6")
    plot_compressed_overview("overview_-4_-6")
    plot_compressed_overview("overview_-2_-6")
    plot_compressed_overview("overview_0_-6")
    plot_compressed_overview("overview_2_-6")
    #plot_compressed_overview("overview_3_-4")
    plot_compressed_overview("overview_4_-3")
    plot_compressed_overview("overview_6_-6")
    plot_compressed_overview("overview_6_-2")
    plot_compressed_overview("overview_6_0")
    plot_compressed_overview("overview_6_2")
    plot_compressed_overview("overview_6_4")
    plt.show()