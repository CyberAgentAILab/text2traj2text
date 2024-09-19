"""Utils for rendering

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import numpy as np

from ..core.layout import Layout

ITEM_COLORS = {
    "": np.array([255, 255, 255]),
    "wall": np.array([100, 100, 100]),
    "item": np.array([150, 150, 150]),
    "casher": np.array([125, 125, 125]),
    "\ufeffwall": np.array([100, 100, 100]),
    "vegetable": np.array([0, 210, 0]),
    "fruit": np.array([255, 106, 248]),
    "fish": np.array([68, 114, 196]),
    "household goods": np.array([255, 252, 0]),
    "seasoning": np.array([169, 209, 141]),
    "snack": np.array([255, 147, 1]),
    "meat": np.array([255, 126, 121]),
    "alcohol": np.array([216, 131, 255]),
    "dairy": np.array([255, 214, 120]),
    "drink": np.array([142, 169, 219]),
    "seafood": np.array([68, 114, 196]),
}


def get_position_color(x: int, y: int, layout: Layout):
    item_name = layout.item_map[y][x]
    if item_name != "" and item_name != "wall" and item_name != "casher":
        category = layout.item_category_dict[item_name]
    else:
        category = item_name

    color = ITEM_COLORS.get(category, ITEM_COLORS["item"]) / 255
    return color
