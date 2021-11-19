from move import sample_levy_aykut, sample_levy
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import levy_stable
from config import Params


def scipy_example():
    fig, ax = plt.subplots(1, 1)
    alpha, beta = 1.5, 0
    mean, var, skew, kurt = levy_stable.stats(alpha, beta, moments='mvsk')

    x = np.linspace(levy_stable.ppf(0.01, alpha, beta),
                    levy_stable.ppf(0.99, alpha, beta), 100)
    ax.plot(x, levy_stable.pdf(x, alpha, beta),
            'r-', lw=5, alpha=0.6, label='levy_stable pdf')

    ax.set_yscale('log')
    ax.set_xscale('log')


def dist_test(title, sample_dist):
    fig, ax = plt.subplots()
    params = Params()
    alphas = [1, 1.5, 2]
    for alpha in alphas:
        params.alpha = alpha
        n_samples = 1000000
        values = sample_dist(n_samples, params)
        space = np.logspace(-2, 5)
        digitized = np.digitize(values, space)
        bin_frequencies = [values[digitized == i].size/n_samples/(space[i]-space[i-1]) for i in range(1, len(space))]

        ax.scatter(space[:-1], bin_frequencies, label=f"alpha = {alpha}")

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative frequency")

if __name__ == '__main__':
    dist_test("Aykut book distribution", sample_levy_aykut)
    dist_test("Abs Symmetric Stable levy distribution", sample_levy)
    dist_test("Shifted Stable levy distribution", lambda x, params: levy_stable.rvs(alpha=params.alpha, beta=1, size=x))
    scipy_example()

    plt.show()