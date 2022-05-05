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


if __name__ == '__main__':
    plot_local_optima()
    plt.show()