from numpy.random import random_sample, randint
from numpy import ndarray
import numpy as np


def two_point_crosser(cross_rate: float):
    return lambda pop: __two_point_crossover(pop, cross_rate)


def __two_point_crossover(parents: ndarray, cross_rate: float) -> ndarray:
    children = np.zeros(parents.shape)
    for i in range(0, len(parents), 2):
        children[i], children[i + 1] = __pair_two_point_crossover(parents[i], parents[i+1], cross_rate)

    return children


def __pair_two_point_crossover(parent1: ndarray, parent2: ndarray, cross_rate: float) -> (ndarray, ndarray):
    child1, child2 = parent1.copy(), parent2.copy()

    if random_sample() < cross_rate:
        # select crossover point that is not on the end of the string
        idx1 = randint(1, len(parent1) - 2)
        idx2 = idx1
        while idx2 == idx1:
            idx2 = randint(1, len(parent1) - 2)

        if idx1 > idx2:
            temp = idx1
            idx1 = idx2
            idx2 = temp

        # perform crossover
        child1[:idx1] = parent1[:idx1]
        child1[idx1:idx2] = parent2[idx1:idx2]
        child1[idx2:] = parent1[idx2:]

        child2[:idx1] = parent2[:idx1]
        child2[idx1:idx2] = parent1[idx1:idx2]
        child2[idx2:] = parent2[idx2:]

    return child1, child2
