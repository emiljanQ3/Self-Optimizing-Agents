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


def plot_local_optima(highlighted=None, old_data=False):
    params = Params()
    params.results_path = "thesis_data/2D_plot/all_combined/" if old_data else "thesis_data/2D_plot/fixed/"
    params.save_id = ""
    results, _ = load_all(params)
    plot.plot_alpha_delta_surface(results, highlighted, old_data)

    save_str = "faulty" if old_data else "fixed"
    plt.savefig(f"figures/opti_{highlighted}_{save_str}.pdf", bbox_inches="tight")


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
    plt.savefig(f"figures/comp_{r0}_{r1}.pdf", bbox_inches="tight")


def plot_genetic_training_history(r0, r1):
    params = Params()
    params.results_path = f"thesis_data/genetic_training/genetic_run_{r0}_{r1}/"
    params.save_id = f"genetic_run_{r0}_{r1}"
    params.tic_rate_0 = r0
    params.tic_rate_1 = r1
    plot.plot_genetic_training_history(params)
    plt.savefig(f"figures/hist_{r0}_{r1}.pdf", bbox_inches="tight")


def plot_validation_distribution(r0, r1):
    params = Params()
    params.results_path = f"thesis_data/overviews/overview_{r0}_{r1}/"
    params.save_id = "gen"
    params.tic_rate_0 = r0
    params.tic_rate_1 = r1

    plot.plot_distribution(params)
    plt.savefig(f"figures/dist_{r0}_{r1}.pdf", bbox_inches="tight")


def setup_matplot_params():
    #plt.rcParams["figure.figsize"] = [4.8, 2.25]  # default [6.4, 4.8]
    pass


def plot_alphas(rs, env=None):
    params = Params()
    params.tic_rate_0, params.tic_rate_1 = rs

    if env is not None:
        params.selected_world = env
        params.results_path = f"thesis_data/initial/??/"
        params.save_id = "???"
        return
    else:
        params.results_path = f"thesis_data/overviews/overview_{rs[0]}_{rs[1]}/"
        params.save_id = ""

    plot.plot_units_over_alpha(params)
    plt.savefig(f"figures/alphas_{rs[0]}_{rs[1]}.pdf", bbox_inches="tight")


if __name__ == '__main__':
    setup_matplot_params()

    plot_local_optima(old_data=True)
    plot_local_optima()
    plot_alphas((0, 0), tags.WorldTag.CONCAVE_CELLS)
    plot_alphas((0, 0), tags.WorldTag.CONVEX_CELLS)
    plot_alphas((0, 0), tags.WorldTag.EMPTY)

    #environments = [
    #                (-4, -6),
    #                (-2, -6),
    #                (0, -6),
    #                (2, -6),
    #                (3, -4),
    #                (4, -3),
    #                (6, -6),
    #                (6, -2),
    #                (6, 0),
    #                (6, 2),
    #                (6, 4)
    #                ]
    #for e in environments:
    #    plot_local_optima(e)
    #    plot_alphas(e)
    #    plot_compressed_overview(*e)
    #    plot_genetic_training_history(*e)
    #    plot_validation_distribution(*e)
    #    pass




    plt.show()