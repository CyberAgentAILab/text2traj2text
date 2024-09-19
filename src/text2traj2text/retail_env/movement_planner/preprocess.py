"""Preprocess trajectory data

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import json
import pathlib
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .core import Layout

current_dir = pathlib.Path(__file__).parent.absolute()
trajectory_dir = (
    current_dir
    / ".."
    / ".."
    / ".."
    / ".."
    / ".."
    / "data"
    / "super-market"
    / "trajectory"
)


def generate_start_and_goals_from_discrete_trajectory(
    trajectory: List[List[int]], layout: Layout
) -> Tuple[NDArray, NDArray]:
    w = 1 / len(layout.floor_map[0])
    h = 1 / len(layout.floor_map)

    start_pos = np.array(
        [
            trajectory[0][0] / len(layout.floor_map) + h / 2,
            trajectory[0][1] / len(layout.floor_map[0]) + w / 2,
        ]
    )

    goal_list = []
    last_stop = False
    last_pos = trajectory[0]
    for pos in trajectory[1:]:
        is_stop = pos[0] == last_pos[0] and pos[1] == last_pos[1]
        if is_stop:
            if last_stop:
                last_stop = True
            else:
                goal_list.append(
                    [
                        pos[0] / len(layout.floor_map) + h / 2,
                        pos[1] / len(layout.floor_map[0]) + w / 2,
                    ]
                )
                last_stop = True

        else:
            last_stop = False
        last_pos = pos
    goal_list.append(
        [
            trajectory[-1][0] / len(layout.floor_map) + h / 2,
            trajectory[-1][1] / len(layout.floor_map[0]) + w / 2,
        ]
    )
    sub_goals = np.array(goal_list)
    return start_pos, sub_goals


def generate_start_and_goals_from_saved_trajectory(
    traj_index: int, layout: Layout
) -> Tuple[NDArray, NDArray]:
    """Generate start and goals from saved trajectory

    Args:
        traj_index (int): trajectory index
        layout (Layout): layout

    Returns:
        Tuple[NDArray, NDArray]: start position and sub goals
    """

    traj_path = trajectory_dir / f"{traj_index}" / "0.json"

    with open(traj_path, "r") as f:
        discrete_trajectory = json.load(f)

    start_pos, sub_goals = generate_start_and_goals_from_discrete_trajectory(
        discrete_trajectory, layout
    )

    return start_pos, sub_goals
