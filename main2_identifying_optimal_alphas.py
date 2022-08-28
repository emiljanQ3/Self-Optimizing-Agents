from config import get_identifying_optimal_alphas_params
from run import run_alpha_r_surface

"""
This main file tests a combination of alpha values and resistance values to see how well different alphas perform for a
given resistance. These results will inform which alpha values are selected in the locally optimal strategy.

A visualization of the performance of different alpha-resistance-combiantions can be seen by running plot2 
identifying_optimal_alphas. Doing so will also print a list of the best found alpha for each resistance. This list 
needs to be copied into the list optimal_alphas in move.py before one can start testing the strategies using the 
optimized alphas.
"""

if __name__ == '__main__':
    params = get_identifying_optimal_alphas_params()
    run_alpha_r_surface(params)