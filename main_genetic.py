import tensorflow as tf
from genetic_keras.genetic_algorithm import genetic_algorithm
from genetic_keras.mutation import StandardMutator
from tensorflow import keras
import pickle
import os
from config import Params
from neural_networks import create_gen_net

params = Params()


def genetic_training(mutation_factor, save_tag, epochs= 1000, decay=1, nr_elites=0):

    save_string = f'saved_genetic_training/{save_tag}/{save_tag}'

    model = create_gen_net(params)
    model.summary()

    model_standard_genetic = keras.models.clone_model(model)

    num_params = model_standard_genetic.count_params()

    standard_genetic_history = genetic_algorithm(model=model_standard_genetic,
                                                 population_size=params.num_agents,
                                                 epochs=epochs,
                                                 mutator=StandardMutator(
                                                     base_mutation_rate=mutation_factor / num_params,
                                                     mutation_decay=decay,
                                                     batches_to_decay=1,
                                                     min_mutation_rate=1 / num_params,
                                                     gene_operation="gauss"),
                                                 initialize_with_model=False,
                                                 nr_elites=nr_elites
                                                 )

    os.makedirs(os.path.dirname(save_string), exist_ok=True)

    with open(save_string + '.pkl', 'wb') as file:
        pickle.dump((standard_genetic_history,), file)

    model_standard_genetic.save(save_string + '_sg.mdl')


if __name__ == '__main__':
    genetic_training(mutation_factor=2, save_tag=params.save_id, epochs=48, decay=1, nr_elites=0)
