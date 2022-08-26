import copy

from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi


class Params:
    def __init__(self, r0=0, r1=0):
        # Meta
        self.num_steps = 100000
        self.num_repeats = 100
        self.num_agents = 10
        self.save_id = "placeholder"
        self.results_path = f"data/placeholder/"
        self.alpha_tag = AlphaInitTag.SAME
        self.model_location = f""
        # Learning
        self.is_genetic_training = False
        self.is_backprop_training = False
        self.update_memory = False
        self.memory_length = 4
        self.memory_compression_factor = 64
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.ang_sd = pi / 6
        self.selected_mover = MoveTag.LEVY_VARYING_DELTA_CONTRAST
        self.alpha = 1  # Only relevant if the current selected strategy does not change alpha
        # World
        self.selected_world = WorldTag.CONCAVE_CELLS
        self.cell_size = 1
        self.area_unit_size = 0.05
        self.obstacle_size = 0.8  # relative to cell
        self.tic_rate_0 = r0  # Resistance in even cells
        self.tic_rate_1 = r1  # Resistance in odd cells
        self.world_height = 2  # Currently only used for initializing the position of the agents
        self.world_width = 2  # Currently only used for initializing the position of the agents
        # Record

        # Saves data for plotting the agent position at each timestep. # Can be used to plot agent trajectories.
        # Will overflow memory quickly, don't use on long runs.
        self.is_recording_position = False
        # Saves the total area discovered for each agent
        self.is_recording_area = False
        # Saves the indices of the are units that are discovered by an agent. Can be used to visualize agent
        # discoveries in combination with trajectories. Will overflow memory quickly, don't use on long runs.
        self.is_recording_area_indices = False
        # Saves the timesteps at which new area was discovered. Makes it possible to visualize area discovered over
        # time and to vary the timespan one examines while still using the same data. I (Emil Jansson) prefer this
        # method over saving the area as a single value.
        self.is_recording_area_over_time = True
        # Records training progress over time.
        self.is_recording_loss = False
        # Collects representative samples from agent memory to use for training outside of simulation.
        # Can be used for deep-Q-learning. This was not used to produce the results of the 2022 thesis.
        self.is_recording_buffer_dataset = False
        # Saves the distribution of effort values selected. Can be used to visualize said
        # distribution in regions with r0 and r1.
        self.is_recording_distribution = False
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


def get_initial_simulations_params():
    params = Params()
    params.save_id = "initial_simulations"
    params.results_path = "data/initial_simulations/"
    return params
