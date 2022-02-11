from tensorflow import keras
import numpy as np
from genetic_keras.data import EpochData


def get_chromosome(model: keras.Model) -> np.ndarray:
    flat_layers = [layer.flatten() for layer in model.get_weights()]
    chromosome = np.concatenate(flat_layers)

    return chromosome


def set_params(model: keras.Model, chromosome: np.ndarray, dummy_params):
    params = []
    i = 0
    for layer in dummy_params:
        ls = layer.size
        flat_layer = chromosome[i:i+ls]
        i += ls
        params.append(np.reshape(flat_layer, layer.shape))

    model.set_weights(params)

# Random batching ignoring the last samples that do not fit in even batches
def random_batching(training_input: np.ndarray, training_output: np.ndarray, batch_size: int):
    p = np.random.permutation(len(training_input))
    training_input = training_input[p]
    training_output = training_output[p]

    batched_input = []
    batched_output = []
    i = 0
    while i + batch_size <= len(training_output):
        batched_input.append(training_input[i:i + batch_size])
        batched_output.append(training_output[i:i + batch_size])
        i += batch_size

    return batched_input, batched_output


def default_print(dataList):
    data: EpochData = dataList[len(dataList) - 1]
    print(f"Epoch {data.index}: completed in {data.time} \n"
          f"Best loss (training, validation): ({data.best_loss_training}, {data.best_loss_validation}). \n"
          f"Median loss (training, validation): ({data.median_loss_training}, {data.median_loss_validation}).")


def accuracy_print(dataList):
    data = dataList[len(dataList) - 1]
    print(f"Epoch {data.index}: completed in {data.time} \n"
          f"Best accuracy (training, validation): ({data.best_accuracy_training}, {data.best_accuracy_validation}). \n"
          f"Median accuracy (training, validation): ({data.median_accuracy_training}, {data.median_accuracy_validation}).")


def mae(y_target, y_pred):
    return np.mean(np.abs(y_pred - y_target))


def logits_int_accuracy(y_target, y_pred):
    pred_labels = np.argmax(y_pred, axis=1)
    diff = pred_labels - np.squeeze(y_target)
    return 1 - np.count_nonzero(diff) / diff.size
