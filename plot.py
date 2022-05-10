import math

import config
import tags
from tags import ResultTag, WorldTag, MoveTag
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from move import cell_exponent
from config import Params
from disk import load_all, quicksave, quickload, load_histories
from genetic_keras.data import EpochData


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

    #plt.show()


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
    ax.legend(["alpha = 1", "alpha = 1.5", "alpha = 2"])


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


def plot_top_contenders(params, force_recalculation=False):
    data, success = quickload(params, "top_contenders")

    best_alpha, best_alpha_data, genetic_data, instant_opti_data, single_alpha_data, slow_opti_data, worst_alpha, \
        worst_alpha_data = data if success and not force_recalculation else prepare_top_contender_data(params)


    fig, ax = plt.subplots()

    if len(single_alpha_data) > 0:
        plot_bar_category(ax, worst_alpha_data, 0)
        plot_bar_category(ax, best_alpha_data, 1)

    if len(slow_opti_data) > 0:
        plot_bar_category(ax, slow_opti_data, 2)

    if len(instant_opti_data) > 0:
        plot_bar_category(ax, instant_opti_data, 3)

    if len(genetic_data) > 0:
        plot_bar_category(ax, genetic_data, 4)

    labels = [f"$\\alpha_{{worst}}: {worst_alpha}$", f"$\\alpha_{{best}}: {best_alpha}$",
              "local", "local$_{s}$", "genetic"]
    ax.bar(range(5), np.zeros(5), tick_label=labels)

    ax.set_title(f"Performance of different strategies when: $r_0 = {params.tic_rate_0}, r_1 = {params.tic_rate_1}$")


def prepare_top_contender_data(params):
    result_list, _ = load_all(params)
    area_times_list = [x[ResultTag.AREA_TIME] for x in result_list]
    mean_areas = []
    for area_times in area_times_list:
        sum = 0
        for at in area_times:
            area = len(at)
            sum += area

        mean = sum / len(area_times)

        mean_areas.append(mean)
    params_list = [x[ResultTag.PARAM] for x in result_list]
    zipped = list(zip(params_list, mean_areas))
    genetic_data = list(filter(lambda it: it[0].selected_mover == MoveTag.DIRECT_TIMER, zipped))
    instant_opti_data = list(
        filter(lambda it: it[0].selected_mover == MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST_INSTANT_SWITCH, zipped))
    slow_opti_data = list(filter(lambda it: it[0].selected_mover == MoveTag.LEVY_OPTIMAL_ALPHA_CONTRAST, zipped))
    single_alpha_data = list(filter(lambda it: it[0].selected_mover == MoveTag.LEVY_VARYING_DELTA_CONTRAST, zipped))
    worst_alpha = "X"
    best_alpha = "X"
    if len(single_alpha_data) > 0:
        best_alpha = single_alpha_data[np.argmax([it[1] for it in single_alpha_data])][0].alpha
        worst_alpha = single_alpha_data[np.argmin([it[1] for it in single_alpha_data])][0].alpha
        best_alpha_data = list(filter(lambda it: it[0].alpha == best_alpha, single_alpha_data))
        worst_alpha_data = list(filter(lambda it: it[0].alpha == worst_alpha, single_alpha_data))

    data = best_alpha, best_alpha_data, genetic_data, instant_opti_data, single_alpha_data, slow_opti_data, worst_alpha, worst_alpha_data
    quicksave(data, params, "top_contenders")
    return data


def plot_bar_category(ax, data, offset):
    cmap = plt.get_cmap("tab10")
    bar_width = 0.7 / len(data) if len(data) != 0 else 1
    ax.bar(np.arange(len(data)) * bar_width + offset - (len(data)-1)/2*bar_width, [it[1] for it in data],
           bar_width, color=cmap(range(len(data))))


def plot_area_in_range(result_list, start_step, end_step, file_names=None, title=""):
    results = [(x[ResultTag.AREA_TIME], x[ResultTag.PARAM]) for x in result_list]

    mean_areas = []
    labels = []
    for area_times, params in results:
        sum = 0
        for at in area_times:
            area = np.sum(np.logical_and(start_step <= np.array(at), np.array(at) <= end_step))
            sum += area

        mean = sum / len(area_times)

        mean_areas.append(mean)
        labels.append(params.save_id)

    if file_names is not None:
        labels = [name[:-4] for name in file_names]

    fig, ax = plt.subplots()

    ax.bar(range(len(mean_areas)), mean_areas, tick_label=labels)

    ax.set_title(title)


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


def plot_alpha_delta_surface(results):
    params = [r[ResultTag.PARAM] for r in results]
    alphas = set()
    deltas = set()
    for p in params:
        p: Params
        alphas.add(p.alpha)
        deltas.add(p.delta_time)

    alphas = list(alphas)
    alphas.sort()
    deltas = list(deltas)
    deltas.sort()
    X = -np.ones((len(deltas), len(alphas)))
    Y = -np.ones((len(deltas), len(alphas)))
    Z = -np.ones((len(deltas), len(alphas)))
    counts = np.zeros((len(deltas), len(alphas)))
    for i in range(len(deltas)):
        for j in range(len(alphas)):
            X[i, j] = deltas[i]/0.5
            Y[i, j] = alphas[j]
            matching_results = filter(lambda it: it[ResultTag.PARAM].alpha == alphas[j]
                                                 and it[ResultTag.PARAM].delta_time == deltas[i], results)

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
            counts[i, j] = count

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(np.log2(X), Y, Z, cmap=cm.get_cmap('viridis'))
    ax.set_xlabel("resistance")
    ax.set_ylabel("alpha")
    ax.set_zlabel("area")
    ax.set_title(f"Optimal alpha for varying resistances. Each point is a mean of {np.min(counts)} simulations.")
    #fig.colorbar(label="area discovered")

    max_indices = np.argmax(Z, axis=1)

    scatter_x = np.log2(X[np.array(range(25)), max_indices])
    scatter_y = Y[np.array(range(25)), max_indices]
    scatter_z = Z[np.array(range(25)), max_indices]
    ax.scatter(scatter_x, scatter_y, scatter_z, c='red')

    string = ""
    for i in range(len(scatter_x)):
        string += str(scatter_x[i]) + ": " + str(scatter_y[i]) + ", "
    print(string)

    fig, ax = plt.subplots()
    ax.pcolor(np.log2(X), Y, Z)
    ax.set_xlabel("resistance")
    ax.set_ylabel("alpha")
    ax.set_title(f"Optimal alpha for varying resistances. Each point is a mean of {np.min(counts)} simulations.")
    ax.scatter(scatter_x, scatter_y, c='red')


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


alpha_list = np.linspace(1, 2, 11)


def __create_memory(random_index, at, params: config.Params):
    memory = np.zeros(params.memory_length)
    for i in range(len(memory)):
        start = random_index - (params.memory_length-i)*params.memory_compression_factor
        end = start + params.memory_compression_factor
        memory[i] = sum(np.logical_and(at >= start, at < end))
    return memory


def generate_memory_examples(result):
    params: config.Params = result[ResultTag.PARAM]
    num_examples = 10
    memory_red_big_delta = np.zeros((len(alpha_list), num_examples, params.memory_length))
    memory_blue_small_delta = np.zeros((len(alpha_list), num_examples, params.memory_length))

    area_times = result[ResultTag.AREA_TIME]
    positions: np.ndarray = result[ResultTag.POSITION][0]

    for i in range(len(alpha_list)):
        at = np.array(area_times[i])
        x_pos = positions[:, i, 0]

        num_reds_saved = 0
        num_blues_saved = 0
        while num_reds_saved < num_examples or num_blues_saved < num_examples:
            random_index = math.floor((np.random.random_sample()*params.num_steps)-params.memory_length*params.memory_compression_factor) + params.memory_length*params.memory_compression_factor

            if np.floor(x_pos[random_index]) % 2 == 0:
                if num_reds_saved >= num_examples:
                    continue
                memory_red_big_delta[i, num_reds_saved] = __create_memory(random_index, at, params)
                num_reds_saved += 1

            else:
                if num_blues_saved >= num_examples:
                    continue
                memory_blue_small_delta[i, num_blues_saved, :] = __create_memory(random_index, at, params)
                num_blues_saved += 1

    return memory_red_big_delta, memory_blue_small_delta


def plot_example_analysis(result):
    area_times = result[ResultTag.AREA_TIME]
    positions: np.ndarray = result[ResultTag.POSITION][0]

    for i in range(len(area_times)):
        at = area_times[i]
        x_pos = positions[:, i, 0]
        at_x_pos = x_pos[at]
        at_color = ["red" if np.floor(x) % 2 == 0 else "blue" for x in at_x_pos]
        pos_color = ["red" if np.floor(x) % 2 == 0 else "blue" for x in x_pos]

        fig, ax = plt.subplots(1, 2)

        plot_world(ax[0], result[ResultTag.PARAM], result[ResultTag.POSITION])
        ax[0].scatter(positions[:, i, 0], positions[:, i, 1], c=pos_color, s=0.5)
        ax[0].set_aspect('equal')

        ax[1].scatter(at, range(len(at)), c=at_color, s=1)

        fig.suptitle("alpha = " + str(alpha_list[i]) + ", Red == large tic, Blue == low tic rate")
        print("wow")
    print("hej")


def plot_single_dist(ax, bin_size, bins, bin_count, title, color):
    x = bins + bin_size/2
    bin_count[1] += bin_count[0]
    y = bin_count[1:]/sum(bin_count)
    ax.bar(x, y, bin_size, color=color)

    ax.set_title(title)
    #ax.set_xlabel("time")
    #ax.set_ylabel("frequency")


def plot_single_dist_log(ax, times, title):
    bins = np.logspace(-3, np.log10(max(times)), 100)
    bin_indices = np.digitize(times, bins)
    bin_count = np.bincount(bin_indices)

    bin_sizes = np.insert(arr=bins[1:] - bins[:-1], obj=0, values=bins[0])
    bin_count = bin_count[:len(bin_sizes)]
    bin_sizes = bin_sizes[:len(bin_count)]

    relative_frequency = bin_count/bin_sizes
    relative_frequency = relative_frequency/sum(relative_frequency)
    x = np.repeat(np.insert(bins, 0, 0), 2)[1:-1]
    y = np.repeat(relative_frequency, 2)
    ax.loglog(x, y)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("frequency")


def plot_distribution(params):
    qs_tag = "distribution"

    data, success = quickload(params, qs_tag)

    if success:
        plot_data_list = data
    else:
        results, filenames = load_all(params)
        plot_data_list = []
        for r in results:
            if tags.ResultTag.DIST in r:
                big, small = r[tags.ResultTag.DIST]

                plot_data = []
                for times in [big, small]:
                    num_bins = 30
                    bin_size = max(times) / num_bins
                    bins = np.linspace(0, max(times), num_bins + 1)
                    bin_indices = np.digitize(times, bins)
                    bin_count = np.bincount(bin_indices)
                    plot_data.append((bin_size, bins, bin_count))

                plot_data_list.append(plot_data)

        quicksave(plot_data_list, params, qs_tag)

    fig, ax = plt.subplots(5, 3)

    cmap = plt.get_cmap("tab10")
    for i in range(len(plot_data_list)):
        big_plot_data = plot_data_list[i][0]
        small_plot_data = plot_data_list[i][1]
        combined_plot_data = (big_plot_data[0], big_plot_data[1], big_plot_data[2]+small_plot_data[2])

        color = cmap(i)
        plot_single_dist(ax[i, 0], *big_plot_data, "$r_0$" if i == 0 else "", color)
        plot_single_dist(ax[i, 1], *small_plot_data, "$r_1$" if i == 0 else "", color)
        plot_single_dist(ax[i, 2], *combined_plot_data, "$combined$" if i == 0 else "", color)

    fig.suptitle(f"$r_0 = {params.tic_rate_0}, r_1 = {params.tic_rate_1}$")
    fig.supxlabel("Frequency")
    fig.supylabel("Time")


def plot_single_cumdist(ax, times, title):
    bins = np.linspace(0, max(times), 100)
    bin_indices = np.digitize(times, bins)
    bin_count = np.bincount(bin_indices)
    bin_count[1] += bin_count[0]
    bin_count = bin_count[1:len(bins)]

    relative_frequency = bin_count / sum(bin_count)
    cumsum = np.cumsum(relative_frequency)
    x = np.repeat(bins, 2)[1:-1]
    y = 1 - np.repeat(np.insert(cumsum, 0, 0), 2)[:-2]
    ax.plot(x, y)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("inverse cumulative frequency")


def plot_single_cumdist_log(ax, times, title):
    bins = np.logspace(-3, np.log10(max(times)), 100)
    bin_indices = np.digitize(times, bins)
    bin_count = np.bincount(bin_indices)
    bin_count = bin_count[:len(bins)]

    relative_frequency = bin_count / sum(bin_count)
    cumsum = np.cumsum(relative_frequency)
    x = np.repeat(np.insert(bins, 0, 0), 2)[1:-1]
    y = 1 - np.repeat(np.insert(cumsum, 0, 0), 2)[:-2]
    ax.loglog(x, y)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("inverse cumulative frequency")


def plot_inverse_cumulative_distribution(results, title=""):
    big_tic_list = []
    small_tic_list = []

    for r in results:
        if tags.ResultTag.DIST in r:
            big, small = r[tags.ResultTag.DIST]
            big_tic_list.extend(big)
            small_tic_list.extend(small)

    fig, ax = plt.subplots(2, 3)

    plot_single_cumdist(ax[0, 0], big_tic_list, "Inverse cumulative distribution in high tic areas")
    plot_single_cumdist(ax[0, 1], small_tic_list, "Inverse cumulative distribution in low tic areas")
    plot_single_cumdist(ax[0, 2], big_tic_list + small_tic_list, "Combined inverse cumulative distribution")

    plot_single_cumdist_log(ax[1, 0], big_tic_list, "Inverse cumulative distribution in high tic areas")
    plot_single_cumdist_log(ax[1, 1], small_tic_list, "Inverse cumulative distribution in low tic areas")
    plot_single_cumdist_log(ax[1, 2], big_tic_list + small_tic_list, "Combined inverse cumulative distribution")

    fig.suptitle(title)


def plot_genetic_training_history(params):
    histories, dir_names = load_histories(params)
    score_histories = []
    for hist in histories:
        score_histories.append([it.median_loss_training for it in hist[0]])

    fig, ax = plt.subplots()
    for i in range(len(score_histories)):
        ax.plot(-np.array(score_histories[i])/0.05**2, label=dir_names[i])

    ax.legend()
    ax.set_xlabel("generations")
    ax.set_ylabel("area units discovered")
