import math
import numpy as np
from tensorflow.keras.models import Model
from tensorflow import keras


# Only for models with a single hidden dense layer
def train_layer(model: Model,
                training_function,
                training_input: np.ndarray,
                training_output: np.ndarray,
                epochs_per_increment: int,
                increment_size: int,
                validation_input: np.ndarray = np.array([]),
                validation_output: np.ndarray = np.array([]),

                ):
    params = model.get_weights()
    layer_size = len(params[1])

    for i in range(len(params)):
        params[i] = params[i] * 0

    histories = []
    num_increments = math.floor(layer_size / increment_size)
    for i in range(num_increments):
        trainable_node_indices = i * increment_size + np.array(range(increment_size))

        histories.append(
            training_function(model, training_input, training_output, validation_input, validation_output,
                              epochs_per_increment, trainable_node_indices)
        )

    return histories


def get_nbn_chromosome(model: keras.Model, node_indices: np.ndarray) -> np.ndarray:
    params = model.get_weights()

    input_hidden_weights = params[0][:, node_indices]
    hidden_biases = params[1][node_indices]
    hidden_output_weights = params[2][node_indices]
    output_biases = params[3]

    chromosome = np.concatenate([np.ravel(input_hidden_weights), hidden_biases,
                                 np.ravel(hidden_output_weights), output_biases])

    return chromosome


def set_nbn_params(model: keras.Model, chromosome: np.ndarray, base_params, node_indices: np.ndarray):
    params = [array.copy() for array in base_params]

    num_hidden = len(node_indices)
    num_inputs = params[0].size // params[1].size
    num_outputs = params[3].size

    i = 0
    params[0][:, node_indices] = np.reshape(chromosome[i:i + num_inputs * num_hidden], (num_inputs, num_hidden))
    i += num_inputs * num_hidden
    params[1][node_indices] = chromosome[i:i + num_hidden]
    i += num_hidden
    params[2][node_indices] = np.reshape(chromosome[i:i + num_hidden * num_outputs], (num_hidden, num_outputs))
    i += num_hidden * num_outputs
    params[3] = chromosome[i:]

    model.set_weights(params)
