from tags import WorldTag, MoveTag, AlphaInitTag
from numpy import pi

class Params:
    def __init__(self):
        # Meta
        self.num_steps = 10000
        self.num_repeats = 1
        self.num_agents = 5
        self.save_id = "Test"
        self.results_path = "results/"
        self.alpha_tag = AlphaInitTag.LINSPACE
        # Movement
        self.delta_time = 1
        self.speed = 1
        self.trans_sd = 0
        self.ang_sd = pi/6
        self.selected_mover = MoveTag.LEVY
        self.alpha = 1
        # World
        self.selected_world = WorldTag.EMPTY
        self.cell_size = 1
        self.obstacle_size = 0.8  # relative to cell
        self.world_height = 10
        self.world_width = 10
        # Record
        self.is_recording_position = True
        # Plot
        self.is_plotting_trajectories = True


