import numpy as np
from tags import ResultTag
import copy


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def normalize_area_to_best_alpha(results):
    max_area_for_speed = dict()
    for r in results:
        max_area_for_speed[r[ResultTag.PARAM].speed] = 0

    for r in results:
        max_area_for_speed[r[ResultTag.PARAM].speed] = max(max_area_for_speed[r[ResultTag.PARAM].speed], np.mean(r[ResultTag.AREA]))

    normalized_results = copy.deepcopy(results)

    for r in normalized_results:
        r[ResultTag.AREA] = r[ResultTag.AREA] / max_area_for_speed[r[ResultTag.PARAM]]

    return normalized_results