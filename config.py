from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 100000
        self.num_repeats = 100
        self.num_agents = 10
        self.save_id = "ReRun2-18_finer_concave"
        self.results_path = "results/rerun2-finer_2D_concave/"
        self.alpha_tag = AlphaInitTag.SAME
        # Movement
        self.delta_time = 0.5
        self.speed = 0.01
        self.trans_sd = 0
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.LEVY
        self.alpha = 1
        # World
        self.selected_world = WorldTag.CONCAVE_CELLS
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.world_height = 1
        self.world_width = 26
        # Record
        self.is_recording_position = False
        self.is_recording_area = True
        self.is_recording_area_indices = False
        self.area_unit_size = 0.05
        # Plot
        self.is_plotting_trajectories = False
        self.is_plotting_area_units = False


