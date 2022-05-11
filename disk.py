import pickle
import os
from config import Params



def save(results, params):
    index = 0
    while True:
        path = params.results_path + params.save_id + str(index) + '.pkl'
        if os.path.exists(path):
            index += 1
        else:
            break

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(results, file)


def load(load_tag, params):
    with open(params.results_path + load_tag + '.pkl', 'rb') as file:
        results = pickle.load(file)

    return results


def load_all_numbered(params):
    index = 0
    results = []
    while True:
        file = params.save_id + str(index) + '.pkl'
        path = params.results_path + file
        if os.path.exists(path):
            with open(path, 'rb') as file:
                results.append(pickle.load(file))
            index += 1
        else:
            break

    return results


def load_all(params: Params):
    results = []
    file_names = []
    for file_name in os.listdir(params.results_path):
        if file_name.startswith(params.save_id):
            file_names.append(file_name)
            try:
                with open(params.results_path + file_name, 'rb') as file:
                    results.append(pickle.load(file))
            except EOFError as e:
                print(f"Failed loading file: {file_name} in {params.results_path}. Possible file corruption.")

    return results, file_names


def load_histories(params: Params):
    histories = []
    dir_names = []
    for dir_name in os.listdir(params.results_path):
        if dir_name.startswith(params.save_id):
            path = params.results_path + dir_name + "/" + dir_name + ".pkl"
            dir_names.append(dir_name)
            try:
                with open(path, 'rb') as file:
                    histories.append(pickle.load(file))
            except EOFError as e:
                print(f"Failed loading file: {path}. Possible file corruption.")

    return histories, dir_names


def map_load_all(params:Params, func):
    results = []
    for file_name in os.listdir(params.results_path):
        if file_name.startswith(params.save_id):
            with open(params.results_path + file_name, 'rb') as file:
                results.append(func(pickle.load(file)))

    return results


def quickload(params, name):
    temp_string: str = params.results_path + params.save_id
    file_name = temp_string.replace("/", "~")
    path = f"quicksaves/{name}-{file_name}"

    try:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        success = True
    except (OSError, IOError) as e:
        data = None
        success = False

    return data, success


def quicksave(data, params, name):
    temp_string: str = params.results_path + params.save_id
    file_name = temp_string.replace("/", "~")
    path = f"quicksaves/{name}-{file_name}"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(data, file)
