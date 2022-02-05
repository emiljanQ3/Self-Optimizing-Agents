import math
import random

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from disk import load_all, map_load_all
from config import Params
from tags import ResultTag
import os
import pickle

params = Params()


def generate_buffers(name, len):
    buffers = map_load_all(params, lambda x : take_first_buffer(x, len))

    path = params.results_path + "buffers_" + name + '.pkl'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(buffers, file)


def generate_dataset(name, test_ratio):
    with open(params.results_path + "buffers_" + name + '.pkl', 'rb') as file:
        buffers = pickle.load(file)

    validation_length = sum([math.floor(len(b)*test_ratio) for b in buffers])
    training_length = sum([len(b) - math.floor(len(b)*test_ratio) for b in buffers])

    train_x = np.zeros((training_length, params.memory_length + 1))
    train_y = np.zeros((training_length, 1))
    val_x = np.zeros((validation_length, params.memory_length + 1))
    val_y = np.zeros((validation_length, 1))

    train_i = 0
    val_i = 0
    for buffer in buffers:
        random.shuffle(buffer)
        num_val = math.floor(len(buffer)*test_ratio)
        for j in range(num_val):
            memory, alpha, reward = buffer[j]
            val_x[val_i, :-1] = memory
            val_x[val_i, -1] = alpha
            val_y[val_i] = reward
            val_i += 1

        for j in range(num_val, len(buffer)):
            memory, alpha, reward = buffer[j]
            train_x[train_i, :-1] = memory
            train_x[train_i, -1] = alpha
            train_y[train_i] = reward
            train_i += 1

    return train_x, train_y, val_x, val_y


def take_first_buffer(result, num):
    buffer = result[ResultTag.BUFFER]
    print(f"Taking {num}/{len(buffer)} at alpha {result[ResultTag.PARAM].alpha}")
    return buffer[:num]


def create_model():
    inputs = layers.Input(shape=(params.memory_length + 1,))
    layer1 = layers.Dense(12, activation="relu")(inputs)
    action = layers.Dense(1, activation="linear")(layer1)
    return keras.Model(inputs=inputs, outputs=action)


if __name__ == '__main__':
    generate_buffers("first_test", 100000)
    train_x, train_y, val_x, val_y = generate_dataset("first_test", 1/10)
    model = create_model()
    model.compile(optimizer="SGD", loss=keras.losses.MSE)
    model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(val_x, val_y))
    model.save("results/model1")