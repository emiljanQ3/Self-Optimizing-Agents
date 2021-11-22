from tags import ResultTag, WorldTag
from matplotlib import pyplot as plt
import numpy as np


def display_results(results, params):

    if params.is_plotting_trajectories:
        if ResultTag.POSITION in results:
            fig, ax = plt.subplots(1, 1)
            if params.is_plotting_area_units:
                if ResultTag.AREA_INDICES in results:
                    plot_area_units(ax, results[ResultTag.AREA_INDICES][0], results[ResultTag.PARAM].area_unit_size)
                else:
                    print("Can't plot area units as there are no area indices saved in results.")

            plot_world(ax, results[ResultTag.PARAM], results[ResultTag.POSITION])
            plot_trajectories(ax, results[ResultTag.POSITION])
        else:
            print("Can't plot position over time as there is no position data saved in results.")

    plt.show()


def plot_world(ax, result_params, result_positions):
    min_x = np.amin(result_positions[:, :, :, 0])
    min_y = np.amin(result_positions[:, :, :, 1])
    max_x = np.amax(result_positions[:, :, :, 0])
    max_y = np.amax(result_positions[:, :, :, 1])

    if result_params.selected_world == WorldTag.CONVEX_CELLS:
        plot_world_convex_cells(ax, max_x, max_y, min_x, min_y, result_params)
    elif result_params.selected_world == WorldTag.CONCAVE_CELLS:
        plot_world_concave_cells(ax, max_x, max_y, min_x, min_y, result_params)


def plot_world_convex_cells(ax, max_x, max_y, min_x, min_y, result_params):
    cell_size = result_params.cell_size
    radius = result_params.obstacle_size * cell_size / 2

    min_x_cell = np.int64(min_x // cell_size)
    min_y_cell = np.int64(min_y // cell_size)
    max_x_cell = np.int64(max_x // cell_size)
    max_y_cell = np.int64(max_y // cell_size)

    for x_cell in range(int(min_x_cell), max_x_cell + 1):
        for y_cell in range(min_y_cell, max_y_cell + 1):
            center_x = x_cell * cell_size + cell_size / 2
            center_y = y_cell * cell_size + cell_size / 2
            poly_x = np.cos(np.linspace(0, 2 * np.pi)) * radius + center_x
            poly_y = np.sin(np.linspace(0, 2 * np.pi)) * radius + center_y

            ax.fill(poly_x, poly_y, 'gray')


def plot_world_concave_cells(ax, max_x, max_y, min_x, min_y, result_params):
    cell_size = result_params.cell_size
    x = result_params.obstacle_size
    h = np.sqrt((np.power(1 - x, 2) + 1) / 4)

    min_x_cell = np.int64(min_x // cell_size)
    min_y_cell = np.int64(min_y // cell_size)
    max_x_cell = np.int64(max_x // cell_size)
    max_y_cell = np.int64(max_y // cell_size)

    for x_cell in range(int(min_x_cell), max_x_cell + 1):
        for y_cell in range(min_y_cell, max_y_cell + 1):
            center_x = x_cell * cell_size + cell_size / 2
            center_y = y_cell * cell_size + cell_size / 2

            x_corners = [0.5, -0.5, -0.5, 0.5]
            y_corners = [0.5, 0.5, -0.5, -0.5]
            for i in range(4):
                unit_poly_x = np.cos(np.linspace(i*np.pi/2, (i+1)*np.pi/2, 200)) * h
                bad_x = np.nonzero(abs(unit_poly_x) > 0.5)
                unit_poly_y = np.sin(np.linspace(i*np.pi/2, (i+1)*np.pi/2, 200)) * h
                bad_y = np.nonzero(abs(unit_poly_y) > 0.5)
                bad = np.append(bad_x, bad_y)

                unit_poly_x = np.delete(unit_poly_x, bad)
                unit_poly_y = np.delete(unit_poly_y, bad)

                unit_poly_x = np.append(unit_poly_x, x_corners[i])
                unit_poly_y = np.append(unit_poly_y, y_corners[i])

                poly_x = unit_poly_x * cell_size + center_x
                poly_y = unit_poly_y * cell_size + center_y

                ax.fill(poly_x, poly_y, 'gray')


def plot_trajectories(ax, positions_over_time):
    ax.plot(positions_over_time[0, :, :, 0], positions_over_time[0, :, :, 1])
    ax.set_aspect('equal')
    ax.legend()


def plot_area_units(ax, indices, area_unit_size):
    for pair in indices:
        base_x = pair[0]*area_unit_size
        base_y = pair[1]*area_unit_size
        x = [base_x, base_x + area_unit_size, base_x + area_unit_size, base_x]
        y = [base_y, base_y, base_y + area_unit_size, base_y + area_unit_size]

        ax.fill(x, y, 'green')


def plot_area_over_alpha(results):
    area = [np.mean(result[ResultTag.AREA]) for result in results]
    alpha = [result[ResultTag.PARAM].alpha for result in results]

    plt.scatter(alpha, area)

    plt.show()
