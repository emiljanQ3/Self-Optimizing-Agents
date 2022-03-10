from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 100000
        self.num_repeats = 100
        self.num_agents = 10
        self.save_id = "gen"
        self.results_path = "results/genetic_run_2_-6_validation/"
        self.alpha_tag = AlphaInitTag.NETWORK
        self.model_location = "saved_genetic_training/genetic_run_2_-6/genetic_run_2_-6"
        # Learning
        self.is_genetic_training = False
        self.is_backprop_training = False
        self.update_memory = True
        self.memory_length = 4
        self.memory_compression_factor = 64
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.DIRECT_TIMER
        self.alpha = 1
        # World
        self.selected_world = WorldTag.CONCAVE_CELLS
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.tic_rate_0 = 2
        self.tic_rate_1 = -6
        self.world_height = 2
        self.world_width = 2
        # Record
        self.is_recording_position = False
        self.is_recording_area = False
        self.is_recording_area_indices = False
        self.is_recording_area_over_time = True
        self.is_recording_loss = False
        self.is_recording_buffer_dataset = False
        self.is_recording_distribution = True
        self.area_unit_size = 0.05
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


