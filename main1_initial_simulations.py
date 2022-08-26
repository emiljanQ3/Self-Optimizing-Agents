from run import run_simulation, run_overview, run_alpha_linspace
from config import get_initial_simulations_params
from tqdm.contrib.concurrent import process_map

"""
This main file creates all data needed for initial plots and saves it in thesis_data/initial.
Run it with command line arguments 0, 1, 2 for concave, convex, empty environments. 
To get all the data you therefore need to run it three times with these three different arguments.

When the program is finished you should see that a folder called data has appeared in your project. Here is the saved 
information from your simulations.
"""

if __name__ == '__main__':
    params = get_initial_simulations_params()
    run_alpha_linspace(params)
