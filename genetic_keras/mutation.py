from numpy.random import random_sample, standard_normal
from numpy import random
import numpy as np
import abc


def __range_flip(old, rand_mult, rand_shift):
    return random_sample() * rand_mult + rand_shift


def __gauss_shift_flip(old, rand_mult, rand_shift):
    return old + standard_normal() * rand_mult


gene_operation_functions = {
    "range": __range_flip,
    "gauss": __gauss_shift_flip
}


class _Mutator(abc.ABC):

    def __init__(self, base_mutation_rate, rand_mult, rand_shift, mutation_decay, batches_to_decay,
                 min_mutation_rate, gene_operation):
        self.count = 0
        self.base_mutation_rate = base_mutation_rate
        self.rand_mult = rand_mult
        self.rand_shift = rand_shift
        self.mutation_decay = mutation_decay
        self.batches_to_decay = batches_to_decay
        self.min_mutation_rate = min_mutation_rate

        self.gene_operation = gene_operation_functions[gene_operation]

        if rand_mult is None:
            if gene_operation == "gauss":
                self.rand_mult = 0.1
            elif gene_operation == "range":
                self.rand_mult = 4
                self.rand_shift = -2

    def mutate(self, population):
        return self.__mutate_population(population)

    def __mutate_population(self, population):
        mutation_rate = max(self.base_mutation_rate * self.mutation_decay ** (self.count // self.batches_to_decay),
                            self.min_mutation_rate)

        mutated_population = np.zeros(population.shape)
        for i in range(len(population)):
            mutated_population[i] = self._single_mutate(population[i], mutation_rate)

        self.count += 1

        return mutated_population

    @abc.abstractmethod
    def _single_mutate(self, chromosome, mutation_rate):
        pass


class StandardMutator(_Mutator):
    def __init__(self, base_mutation_rate, rand_mult=None, rand_shift=None, mutation_decay=1, batches_to_decay=1,
                 min_mutation_rate=0, gene_operation="range"):
        super().__init__(base_mutation_rate, rand_mult, rand_shift, mutation_decay, batches_to_decay, min_mutation_rate,
                         gene_operation)

    def _single_mutate(self, chromosome, mutation_rate):
        num_mutations = random.binomial(len(chromosome), mutation_rate)
        indices = random.choice(len(chromosome), num_mutations)
        for i in indices:
            chromosome[i] = self.gene_operation(chromosome[i], self.rand_mult, self.rand_shift)

        return chromosome

