from numpy.random import randint, random_sample
import numpy as np

from genetic_keras import crossover, mutation, selection, utils, data, evaluation, nbn_training
from tensorflow import keras
import time
from matplotlib import pyplot as plt
import math
from tqdm import tqdm


def _eliteism_insertion(mutated_children, population, losses, nr_elites):
    index_order = np.argsort(losses)
    elites = population[index_order[0:nr_elites]]
    mutated_children[0:nr_elites] = elites
    return mutated_children


def genetic_algorithm(model: keras.Model,
                      epochs,
                      population_size=100,
                      mutator=None,
                      f_cross=crossover.two_point_crosser(0.9),
                      f_select=selection.tournament_selector(2, 0.75),
                      f_plot=None,
                      f_print=utils.default_print,
                      initialize_with_model=True,
                      keep_interval=100,
                      nr_elites=0
                      ):
    # Initialization
    model_chromosome = utils.get_chromosome(model)

    n_genes = model_chromosome.size

    if mutator is None:
        mutator = mutation.StandardMutator(1 / n_genes)

    if f_plot is not None:
        fig, ax = plt.subplots(1, 3)
        plt.show()

    population = random_sample((population_size, n_genes)) * 4 - 2 # Weight init is hard-coded atm
    if initialize_with_model:
        for i in range(len(population)):
            population[i] = model_chromosome

    epoch_history = []

    # Algorithm
    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        losses = evaluation.evaluate_population(population, model, f_loss, batched_input[batch], batched_output[batch], node_indices)

        parents = f_select(population, losses)

        children = f_cross(parents)

        mutated_children = mutator.mutate(children)

        next_generation = _eliteism_insertion(mutated_children, population, losses, nr_elites)

        population = next_generation

        # End of epoch

        epoch_time = time.time()-start_time
        epoch_history.append(data.generate_epoch_data(population=population,
                                                      train_losses=None,
                                                      validation_losses=None,
                                                      epoch_time=epoch_time,
                                                      index=epoch,
                                                      train_accuracy=None,
                                                      validation_accuracy=None))

        # Only keep every keep_interval chromosomes due to memory constraints
        if epoch-1 % keep_interval != 0 and epoch != 0:
            epoch_history[epoch-1].best_chromosome_training = None
            epoch_history[epoch-1].best_chromosome_validation = None

        if f_print is not None:
            f_print(epoch_history)

    return epoch_history, best
