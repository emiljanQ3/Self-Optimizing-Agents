from enum import Enum


class MoveTag(Enum):
    BROWNIAN = "brownian"
    ACTIVE_ROTDIFF = "active_rotdiff"
    AYKUT_LEVY = "aykut_levy"
    LEVY = "levy"
    LEVY_VARYING_DELTA = "levy_varying_delta"
    LEVY_OPTIMAL_ALPHA = "levy_optimal_alpha"
    LEVY_VARYING_DELTA_CONTRAST = "levy_varying_delta_contrast"
    LEVY_OPTIMAL_ALPHA_CONTRAST = "levy_optimal_alpha_contrast"
    LEVY_OPTIMAL_ALPHA_CONTRAST_INSTANT_SWITCH = "LEVY_OPTIMAL_ALPHA_CONTRAST_INSTANT_SWITCH"
    NEURAL_LEVY = "neural_levy"
    DIRECT_TIMER = "direct_timer"


class WorldTag(Enum):
    EMPTY = "empty"
    EMPTY_REPEATING = "empty_repeating"
    CONVEX_CELLS = "convex_cells"
    CONCAVE_CELLS = "concave_cells"


class ResultTag:
    POSITION = "position"
    PARAM = "param"
    AREA = "area"
    AREA_INDICES = "area_indices"
    AREA_TIME = "area_time"
    LOSS = "loss"
    BUFFER = "buffer"


class AlphaInitTag:
    LINSPACE = "linspace"
    SAME = "same"
    NETWORK = "network"
