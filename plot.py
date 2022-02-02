import math

from tags import ResultTag, WorldTag, MoveTag
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from move import cell_exponent


def display_results(results, params):
    result_params = results[ResultTag.PARAM]
    if params.is_plotting_trajectories:
        if ResultTag.POSITION in results:
            fig, ax = plt.subplots(1, 1)
            if params.is_plotting_area_units:
                if ResultTag.AREA_INDICES in results:
                    plot_area_units(ax, results[ResultTag.AREA_INDICES][0], result_params.area_unit_size)
                else:
                    print("Can't plot area units as there are no area indices saved in results.")

            plot_world(ax, result_params, results[ResultTag.POSITION])

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

    if result_params.selected_mover == MoveTag.LEVY_OPTIMAL_ALPHA \
            or result_params.selected_mover == MoveTag.LEVY_VARYING_DELTA:
        plot_varying_delta(ax, result_params, min_x, max_x)


def plot_world_convex_cells(ax, max_x, max_y, min_x, min_y, result_params):
    cell_size = result_params.cell_size
    radius = result_params.obstacle_size * cell_size / 2

    min_x_cell = -3  # np.int64(min_x // cell_size)
    min_y_cell = -3  # np.int64(min_y // cell_size)
    max_x_cell = 2  # np.int64(max_x // cell_size)
    max_y_cell = 2  # np.int64(max_y // cell_size)

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
                unit_poly_x = np.cos(np.linspace(i * np.pi / 2, (i + 1) * np.pi / 2, 200)) * h
                bad_x = np.nonzero(abs(unit_poly_x) > 0.5)
                unit_poly_y = np.sin(np.linspace(i * np.pi / 2, (i + 1) * np.pi / 2, 200)) * h
                bad_y = np.nonzero(abs(unit_poly_y) > 0.5)
                bad = np.append(bad_x, bad_y)

                unit_poly_x = np.delete(unit_poly_x, bad)
                unit_poly_y = np.delete(unit_poly_y, bad)

                unit_poly_x = np.append(unit_poly_x, x_corners[i])
                unit_poly_y = np.append(unit_poly_y, y_corners[i])

                poly_x = unit_poly_x * cell_size + center_x
                poly_y = unit_poly_y * cell_size + center_y

                ax.fill(poly_x, poly_y, 'gray')


def plot_varying_delta(ax, result_params, min_x, max_x):
    x = np.linspace(min_x, max_x, math.ceil(max_x-min_x)*100)
    y = cell_exponent(x, result_params.world_width)

    ax.plot(x, y)


def plot_trajectories(ax, positions_over_time):
    ax.plot(positions_over_time[0, :, :, 0], positions_over_time[0, :, :, 1])
    ax.set_aspect('equal')
    ax.legend()


def plot_area_units(ax, indices, area_unit_size):
    for pair in indices:
        base_x = pair[0] * area_unit_size
        base_y = pair[1] * area_unit_size
        x = [base_x, base_x + area_unit_size, base_x + area_unit_size, base_x]
        y = [base_y, base_y, base_y + area_unit_size, base_y + area_unit_size]

        ax.fill(x, y, 'green')


def plot_area_over_alpha(results):
    area = [np.mean(result[ResultTag.AREA]) for result in results]
    max_area = np.max(area)
    area = [x / max_area for x in area]
    alpha = [result[ResultTag.PARAM].alpha for result in results]

    num_repeats = results[0][ResultTag.PARAM].num_repeats
    num_agents = results[0][ResultTag.PARAM].num_agents

    fig, ax = plt.subplots()
    ax.scatter(alpha, area)
    ax.set_title(f"Explored area, mean over {num_repeats * num_agents} agents.")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Normalized explored area.")
    plt.show()


def plot_last_area_over_alpha(result_list, last_steps):
    results = [(x[ResultTag.AREA_TIME], x[ResultTag.PARAM]) for x in result_list]

    mean_areas = []
    labels = []
    for area_times, params in results:
        sum = 0
        for at in area_times:
            area = np.sum(np.array(at) > (params.num_steps - last_steps))
            sum += area

        mean = sum / len(area_times)

        mean_areas.append(mean)
        labels.append(params.save_id)

    fig, ax = plt.subplots()

    ax.bar(range(len(mean_areas)), mean_areas, tick_label=labels)


def plot_many_area_at_time(result_list, time):
    fig, ax = plt.subplots()
    for result in result_list:
        params = result[ResultTag.PARAM]
        label = "alpha: " + str(params.alpha) if len(params.alpha_times) == 0 else "varying alpha"
        max_area = 20 ** 2 * np.pi

        plot_area_at_time(ax, result, time, max_area=max_area, label=label)

    ax.set_title(f"Explored area at time = {time}, mean over {params.num_repeats * params.num_agents} agents.")
    ax.set_ylabel("Normalized explored area.")
    plt.draw()


def plot_area_at_time(ax, results, target_time, max_area, label):
    area, time = get_area_time(max_area, results)

    index = np.argmax(np.array(time) >= target_time)

    ax.bar(label, area[index])


def plot_many_area_over_time(result_list, time=None):
    fig, ax = plt.subplots()

    if time is not None:
        ax.plot([time, time], [0, 1], 'k')

    i = 0
    for result in result_list:
        i += 1
        params = result[ResultTag.PARAM]
        label = "alpha: " + str(params.alpha) if len(params.alpha_times) == 0 else "varying alpha"
        line = "--" if i > 10 else "-"
        max_area = 20 ** 2 * np.pi

        plot_area_over_time(ax, result, max_area=max_area, label=label, line=line)

    ax.set_title(f"Explored area, mean over {params.num_repeats * params.num_agents} agents.")
    ax.set_xlabel("time")
    ax.set_ylabel("Normalized explored area.")
    ax.legend()
    plt.draw()


def plot_area_over_time(ax, results, max_area, label, line):
    area, time = get_area_time(max_area, results)

    ax.plot(time, area, line, label=label)


def get_area_time(max_area, results):
    params = results[ResultTag.PARAM]
    area = get_area_over_steps(max_area, results)
    if len(params.viscosity_times) > 0:
        time = get_varied_time(params)
    else:
        time = [step * params.delta_time for step in range(params.num_steps)]
    return area, time


def get_area_over_steps(max_area, results):
    params = results[ResultTag.PARAM]
    area_times = results[ResultTag.AREA_TIME]
    area = np.zeros(params.num_steps)
    for at in area_times:
        for step in at:
            area[step:] += 1
    area = area / max_area / len(area_times)
    return area


def get_varied_time(params):
    time = []
    clock = 0
    timer = 0
    i = 0
    base_speed = params.speed
    base_delta_t = 0.5  # TODO revert params.delta_time
    for step in range(params.num_steps):
        if timer == 0:
            speed_factor, timer = params.viscosity_times[i]
            delta_time = base_delta_t / speed_factor

            i = (i + 1) % len(params.viscosity_times)

        clock += delta_time
        timer -= 1
        time.append(clock)
    return time


def plot_alpha_speed_surface(results):
    params = [r[ResultTag.PARAM] for r in results]
    alphas = set()
    speeds = set()
    for p in params:
        alphas.add(p.alpha)
        speeds.add(p.speed)

    alphas = list(alphas)
    alphas.sort()
    speeds = list(speeds)
    speeds.sort()
    X = -np.ones((len(speeds), len(alphas)))
    Y = -np.ones((len(speeds), len(alphas)))
    Z = -np.ones((len(speeds), len(alphas)))
    for i in range(len(speeds)):
        for j in range(len(alphas)):
            X[i, j] = speeds[i]
            Y[i, j] = alphas[j]
            matching_results = filter(lambda it: it[ResultTag.PARAM].alpha == alphas[j]
                                                 and it[ResultTag.PARAM].speed == speeds[i], results)

            count = 0
            sum = 0
            for r in matching_results:
                areas = r[ResultTag.AREA]
                sum += np.sum(areas)
                count += np.size(areas)

            if count == 0:
                mean = -1
            else:
                mean = sum / count

            Z[i, j] = mean

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(np.log2(X), Y, Z, cmap=cm.get_cmap('viridis'))
    ax.set_xlabel("log2(speed)")
    ax.set_ylabel("alpha")
    ax.set_zlabel("area")


def scatter_alpha_speed_surface(results):
    params = [r[ResultTag.PARAM] for r in results]
    areas = [r[ResultTag.AREA] for r in results]

    x = [np.log2(it.speed) for it in params]
    y = [it.alpha for it in params]
    z = [np.mean(it) for it in areas]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z)
    ax.set_xlabel("log2(speed)")
    ax.set_ylabel("alpha")
    ax.set_zlabel("area")


def plot_loss_over_time(results, rolling_mean=1):
    loss_collections = [r[ResultTag.LOSS] for r in results]
    agent_losses = [item for sublist in loss_collections for item in sublist]

    if rolling_mean > 1:
        agent_losses = [fast_moving_average(al, rolling_mean) for al in agent_losses]

    fig, ax = plt.subplots()
    for al in agent_losses:
        ax.plot(al)


def moving_average(list, n):
    averaged = np.zeros(len(list)+n)
    for i in range(len(list)):
        if list[i] is not None:

            averaged[i:i+n] += list[i]
    averaged /= n
    return averaged[n-1:-n]


def fast_moving_average(list, n):
    ret = np.cumsum([i for i in list if i], dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n