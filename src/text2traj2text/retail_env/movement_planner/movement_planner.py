"""Combined planner using PRM and DWA

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from chex import dataclass

from ..core.core import Position, State
from .core import Layout
from .dwa import DWA
from .prm import PRM


@dataclass
class MovementPlanner:
    layout: Layout
    prm_args: dict = None
    dwa_args: dict = None
    goal_radius: float = (0.01,)
    stop_duration: int = (4,)

    def __post_init__(self):
        if self.prm_args is not None:
            self.global_planner = PRM(layout=self.layout, **self.prm_args)
        else:
            self.global_planner = PRM(layout=self.layout)
        if self.dwa_args is not None:
            self.local_planner = DWA(layout=self.layout, **self.dwa_args)
        else:
            self.local_planner = DWA(layout=self.layout)

    def plan(
        self,
        start_pos: Position,
        start_r: float,
        goal_pos: Position,
        goal_radius: float,
    ) -> Tuple[State, Position]:
        """
        Plan a trajectory by the combination of PRM and DWA

        Args:
            start_pos (NDArray): start position
            start_r (float): start rotation
            goal_pos (NDArray): goal position
            goal_radius (float): radius of goal region

        Returns:
            Tuple[State, Position]: trajectory and path
        """

        path = self.global_planner.plan(start_pos, goal_pos)

        traj = []
        start_v, start_a = 0.0, 0.0
        for i in range(len(path) - 1):
            traj_wp = self.local_planner.plan(
                State(pos=start_pos, r=start_r, v=start_v, a=start_a),
                path.at(i + 1),
                goal_radius=goal_radius,
            )
            traj.append(traj_wp)
            start_pos = Position.from_array(traj_wp[-1, :2])
            start_r = traj_wp[-1, 2]
            start_v = traj_wp[-1, 3]
            start_a = traj_wp[-1, 4]

        traj = np.vstack(traj)
        traj = State(
            pos=Position(traj[:, 0], traj[:, 1]),
            r=traj[:, 2],
            v=traj[:, 3],
            a=traj[:, 4],
        )

        return traj, path
