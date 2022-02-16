from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 100000
        self.num_repeats = 1
        self.num_agents = 11
        self.save_id = "run_linspace"
        self.results_path = "results/inspecting_data/"
        self.alpha_tag = AlphaInitTag.LINSPACE
        self.model_location = "saved_genetic_training/gentic_run_3-/gentic_run_3-"
        # Learning
        self.is_genetic_training = False
        self.is_backprop_training = False
        self.update_memory = True
        self.memory_length = 16
        self.memory_compression_factor = 16
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.LEVY_VARYING_DELTA_CONTRAST
        self.alpha = 1
        # World
        self.selected_world = WorldTag.CONCAVE_CELLS
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.world_height = 2
        self.world_width = 2
        # Record
        self.is_recording_position = True
        self.is_recording_area = False
        self.is_recording_area_indices = False
        self.is_recording_area_over_time = True
        self.is_recording_loss = False
        self.is_recording_buffer_dataset = False
        self.area_unit_size = 0.05
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


