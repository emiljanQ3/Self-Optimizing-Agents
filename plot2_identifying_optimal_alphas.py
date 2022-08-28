import matplotlib.pyplot as plt

import config
from disk import load_all
from plot import plot_alpha_delta_surface

if __name__ == '__main__':
    params = config.get_identifying_optimal_alphas_params()
    results, _ = load_all(params)

    plot_alpha_delta_surface(results)

    plt.savefig(f"figures/identifying_optimal_alphas.pdf", bbox_inches="tight")
    plt.show()
