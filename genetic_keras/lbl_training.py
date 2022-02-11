import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Input
from itertools import takewhile


def train_network(model: Model,
                  training_function,
                  training_input: np.ndarray,
                  training_output: np.ndarray,
                  validation_input: np.ndarray = np.array([]),
                  validation_output: np.ndarray = np.array([]),
                  epochs: list = 1000,
                  ):
    network_depth = len(model.layers)

    if not isinstance(epochs, list):
        epochs = [epochs for x in range(network_depth - 1)]

    histories = []
    saved_weights = []
    current_layer = None
    k = 0
    for i in range(0, network_depth - 1):

        if not model.layers[i].trainable_weights:
            continue

        prev_layers = Sequential()
        for j in range(i):
            prev_layers.add(model.layers[j])
        prev_layers.set_weights(saved_weights)
        print("Prev layers")
        prev_layers.summary()

        current_layer = Sequential()
        current_layer.add(Input(shape=prev_layers.output_shape[1:]))
        current_layer.add(model.layers[i])
        current_layer.add(Dense(
            model.output_shape[1]))
        print("Current layer")
        current_layer.summary()

        current_training_input = prev_layers(training_input).numpy()
        current_validation_input = prev_layers(validation_input).numpy()

        maybe_history = training_function(current_layer, current_training_input, training_output,
                                          current_validation_input, validation_output, epochs[k])

        k += 1

        saved_weights.append(current_layer.get_weights()[0])
        saved_weights.append(current_layer.get_weights()[1])
        histories.append(maybe_history)

    saved_weights.append(current_layer.get_weights()[2])
    saved_weights.append(current_layer.get_weights()[3])

    model.set_weights(saved_weights)
    return histories
