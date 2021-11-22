from enum import Enum

class MoveTag(Enum):
    BROWNIAN = "brownian"
    ACTIVE_ROTDIFF = "active_rotdiff"
    AYKUT_LEVY = "aykut_levy"
    LEVY = "levy"


class WorldTag(Enum):
    EMPTY = "empty"
    EMPTY_REPEATING = "empty_repeating"
    CONVEX_CELLS = "convex_cells"
    CONCAVE_CELLS = "concave_cells"


class ResultTag:
    POSITION = "position"
    PARAM = "param"
    AREA = "area"


class AlphaInitTag:
    LINSPACE = "linspace"
    SAME = "same"
