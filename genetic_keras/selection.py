import numpy as np
from numpy.random import randint, random_sample


def tournament_selector(k, p_tour):
    return lambda pop, losses: __tournament_selection(pop, losses, k, p_tour)


def __tournament_selection(pop: np.ndarray, losses, k, p_tour) -> np.ndarray:
    new_pop = np.zeros(pop.shape)
    for i in range(len(pop)):
        new_pop[i] = __single_tournament_selection(pop, losses, k, p_tour)

    return new_pop


def __single_tournament_selection(pop: np.ndarray, losses, k, p_tour) -> np.ndarray:
    candidate_indexes = randint(0, len(pop), k)
    candidate_losses = losses[candidate_indexes]
    sorted_candidates_indexes = candidate_indexes[np.argsort(candidate_losses)]

    for i in range(k-1):
        if random_sample() < p_tour:
            return pop[sorted_candidates_indexes[i]]

    return pop[sorted_candidates_indexes[k-1]]
