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


class WorldTag(Enum):
    EMPTY = "empty"
    EMPTY_REPEATING = "empty_repeating"
    CONVEX_CELLS = "convex_cells"
    CONCAVE_CELLS = "concave_cells"
    CIRCLE = "circle"


class ResultTag:
    POSITION = "position"
    PARAM = "param"
    AREA = "area"
    AREA_INDICES = "area_indices"


class AlphaInitTag:
    LINSPACE = "linspace"
    SAME = "same"
