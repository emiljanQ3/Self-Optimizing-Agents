import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import Params
import copy


# Configuration paramaters for the whole setup
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_decay_factor = 0.99
initial_actions_before_training = 1000
max_replay_buffer_size = np.inf
batch_size = 32  # Size of batch taken from replay buffer
compressed_memory_length = 16
memory_compression_rate = 16
num_alpha_options = 11

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.MeanSquaredError()


class NeuralNetworkContainer:

    def __init__(self, params: Params):
        self.epsilon = epsilon_max
        
        self.experience_buffer = []
        self.network = create_qnet(params)
        # self.network_copy = copy.deepcopy(self.network)
        self.last_memory = None
        self.last_selected_alpha = None

    def get_next_alpha(self, recent_memory, mean_reward):
        compressed_memory = compress(recent_memory)
        selected_alpha = self.__select_alpha(compressed_memory)

        if self.last_memory is not None:
            self.experience_buffer.append((self.last_memory, self.last_selected_alpha, mean_reward))
            if len(self.experience_buffer) > max_replay_buffer_size:
                self.experience_buffer.remove(0)

        self.last_memory = compressed_memory
        self.last_selected_alpha = selected_alpha

        return selected_alpha

    def train_network(self, num_batches):
        if len(self.experience_buffer) < initial_actions_before_training:
            return

        for i in range(num_batches):
            chosen_experiences = np.random.choice(self.experience_buffer, batch_size, replace=False)
            compressed_memories = np.array([it[0] for it in chosen_experiences])
            alpha = np.array([it[1] for it in chosen_experiences])
            reward = np.array([it[2] for it in chosen_experiences])

            x = np.zeros((batch_size, compressed_memory_length + 1))
            x[:, :-1] = compressed_memories
            x[:, -1] = alpha
            y_target = reward

            with tf.GradientTape() as tape:
                y_actual = self.network(x, training=True)
                loss_value = loss_fn(y_target, y_actual)

            grads = tape.gradient(loss_value, self.network.trainable_weights)

            optimizer.apply_gradients(zip(grads, self.network.trainable_weights))

    def __select_alpha(self, compressed_memory):
        r = np.random.random_sample()
        if r < self.epsilon:
            selected_alpha = np.random.random_sample() + 1
        else:
            selected_alpha = self.__network_select(compressed_memory)
            
        self.__epsilon_decay()
        return selected_alpha

    def __network_select(self, compressed_memory):
        alpha_options = np.linspace(1, 2, num_alpha_options)

        base_row = np.append(compressed_memory, 0)
        batch = np.tile(base_row, (num_alpha_options, 1))
        batch[:, -1] = alpha_options

        results = self.network(batch)

        return alpha_options[np.argmax(results)]

    def __epsilon_decay(self):
        if len(self.experience_buffer) > initial_actions_before_training:
            self.epsilon *= epsilon_decay_factor
            if self.epsilon < epsilon_min:
                self.epsilon = epsilon_min


def create_qnet(params: Params):
    inputs = layers.Input(shape=(compressed_memory_length + 1,))

    layer1 = layers.Dense(12, activation="relu")(inputs)

    action = layers.Dense(1, activation="linear")(layer1)

    return keras.Model(inputs=inputs, outputs=action)


def compress(uncompressed_memory):
    compressed_memory = np.zeros(compressed_memory_length)

    for i in range(compressed_memory_length):
        compressed_memory[i] = np.sum(uncompressed_memory[i*memory_compression_rate:(i+1)*memory_compression_rate])

    return compressed_memory
