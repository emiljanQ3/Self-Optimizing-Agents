from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 100000
        self.num_repeats = 10
        self.num_agents = 100
        self.save_id = "Run10_medium_Convex_world"
        self.results_path = "results/"
        self.alpha_tag = AlphaInitTag.SAME
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.trans_sd = 0
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.LEVY
        self.alpha = 1
        # World
        self.selected_world = WorldTag.CONVEX_CELLS
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.world_height = 10
        self.world_width = 10
        # Record
        self.is_recording_position = False
        self.is_recording_area = True
        self.is_recording_area_indices = False
        self.area_unit_size = 0.05
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


