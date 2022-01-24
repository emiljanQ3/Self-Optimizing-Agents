from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 10000000
        self.num_repeats = 1
        self.num_agents = 10
        self.save_id = "run39"
        self.results_path = "results/neural_training_1/"
        self.alpha_tag = AlphaInitTag.NETWORK
        # Learning
        self.train_network = True
        self.update_memory = True
        self.memory_length = 16
        self.memory_compression_factor = 16
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.NEURAL_LEVY
        self.alpha = 1
        # World
        self.selected_world = WorldTag.CONCAVE_CELLS
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.world_height = 2
        self.world_width = 2
        # Record
        self.is_recording_position = False
        self.is_recording_area = True
        self.is_recording_area_indices = False
        self.is_recording_area_over_time = False
        self.area_unit_size = 0.05
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


