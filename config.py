from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 50000
        self.num_repeats = 100
        self.num_agents = 10
        self.save_id = "run36"
        self.results_path = "results/changing_circle_high_contrast/"
        self.alpha_tag = AlphaInitTag.SAME
        self.alpha_times = [(2, 1000), (1, 1000)]
        self.redirect_on_alpha_change = True
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.trans_sd = 0
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.LEVY
        self.alpha = 1
        # World
        self.selected_world = WorldTag.CIRCLE
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.world_height = 2
        self.world_width = 2
        self.viscosity_times = [(2**6, 1000), (2**(-6), 1000)]
        # Record
        self.is_recording_position = False
        self.is_recording_area = False
        self.is_recording_area_indices = False
        self.is_recording_area_over_time = True
        self.area_unit_size = 0.05
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


