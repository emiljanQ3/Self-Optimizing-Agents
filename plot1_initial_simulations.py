import config
import plot
import tags
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for env in [tags.WorldTag.CONCAVE_CELLS, tags.WorldTag.CONVEX_CELLS, tags.WorldTag.EMPTY]:
        params = config.get_initial_simulations_params()
        params.selected_world = env
        params.save_id += f"_{str(params.selected_world)[9:]}"

        fig = plot.plot_units_over_alpha(params)


        fig.set_size_inches(3.2, 2.4)  # default [6.4, 4.8]
        string = "convex" if env == tags.WorldTag.CONVEX_CELLS else \
            "concave" if env == tags.WorldTag.CONCAVE_CELLS else "homogenous"
        plt.savefig(f"figures/initial_{string}.pdf", bbox_inches="tight")

    plt.show()