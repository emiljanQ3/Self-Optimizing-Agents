from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

r0 = 0
r1 = 0

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 100000
        self.num_repeats = 100
        self.num_agents = 10
        self.save_id = ""
        self.results_path = f"thesis_data/overviews/opti_rerun"
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
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.LEVY_VARYING_DELTA_CONTRAST
        self.alpha = 1  # Only relevant if the current selected strategy does not change alpha
        # World
        self.selected_world = WorldTag.CONCAVE_CELLS
        self.cell_size = 1
        self.area_unit_size = 0.05
        self.obstacle_size = 0.8  # relative to cell
        self.tic_rate_0 = r0   # Resistance in even cells
        self.tic_rate_1 = r1   # Resistance in odd cells
        self.world_height = 2   # Currently only used for initializing the position of the agents
        self.world_width = 2   # Currently only used for initializing the position of the agents
        # Record
        self.is_recording_position = False
        self.is_recording_area = False
        self.is_recording_area_indices = False
        self.is_recording_area_over_time = True
        self.is_recording_loss = False
        self.is_recording_buffer_dataset = False
        self.is_recording_distribution = False
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


