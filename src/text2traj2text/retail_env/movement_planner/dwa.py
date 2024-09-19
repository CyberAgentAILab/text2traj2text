"""DWA (Dynamic Window Approach) local planner.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo

Reference: https://github.com/CyberAgentAILab/retail-env
dwa implementation is based on the retail-env library.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import dataclass

from ..core.core import Obstacle, Position, State
from .core import Action, BaseLocalPlanner


def compute_obstacle_distance(pos: Position, obs: Obstacle):
    def _distance(pos: Position, obs: Obstacle):
        distance_x = (pos.x < obs.x) * (
            pos.x - obs.x
        )  # x distance if pos is left of obs
        distance_x += (pos.x > (obs.x + obs.w)) * (
            pos.x - (obs.x + obs.w)
        )  # x distance if pos is right of obs
        distance_x += (
            jnp.logical_and((obs.x <= pos.x), (pos.x < (obs.x + obs.w))) * 100
        )  # x distance if pos is inside obs from x axis view point

        distance_y = (pos.y < obs.y) * (obs.y - pos.y)  # y distance if pos is below obs
        distance_y += (pos.y > (obs.y + obs.h)) * (pos.y - (obs.y + obs.h))
        distance_y += (
            jnp.logical_and((obs.y <= pos.y), (pos.y < (obs.y + obs.h))) * 100
        )  # y distance if pos is inside obs from y axis view point

        return jnp.maximum(distance_x, distance_y)

    distances = jax.vmap(_distance, in_axes=(0, None))(pos, obs)
    distances = jnp.min(distances, axis=1)
    return distances


@dataclass
class DWA(BaseLocalPlanner):
    vel_decay: float = 0.05
    vel_res: int = 16
    ang_res: int = 16
    # obstacle_weight: float = 0

    def _build_act(self):
        def act(state: State, goal_pos: Position) -> Action:
            """
            Select an action based on a simplified DWA algorithm.
            The action that moves the agent closest to the goal is selected.

            Args:
                state (State): current state
                goal_pos (Array): goal position

            Returns:
                Action: selected action
            """
            vel_weight = 1 - jnp.exp(
                -jnp.linalg.norm(
                    jnp.array([goal_pos.x - state.pos.x, goal_pos.y - state.pos.y])
                    / self.vel_decay
                )
            )

            v_cands = (
                jnp.linspace(
                    jnp.clip(state.v - self.max_acc, a_min=0),
                    jnp.clip(state.v + self.max_acc, a_max=self.max_vel),
                    self.vel_res,
                ).flatten()
                * vel_weight
                / self.max_vel
            )
            a_cands = (
                jnp.linspace(
                    jnp.clip(state.a - self.max_ang_acc, a_min=-self.max_ang),
                    jnp.clip(state.a + self.max_ang_acc, a_max=self.max_ang),
                    self.ang_res,
                ).flatten()
                / self.max_ang
            )
            dwin = jnp.vstack([x.flatten() for x in jnp.meshgrid(v_cands, a_cands)]).T
            next_states = jax.vmap(
                lambda s, ar: self.compute_next_state(s, Action(nv=ar[0], na=ar[1])),
                in_axes=(None, 0),
            )(state, dwin)
            validity = jax.vmap(self.layout.validate_line, in_axes=(None, 0))(
                state.pos, next_states.pos
            )
            goal_dists = jnp.linalg.norm(
                next_states.pos.array() - goal_pos.array(), axis=-1
            )
            score = -goal_dists

            score = score * validity + -jnp.inf * ~validity

            selected_idx = jnp.argmax(score)
            nv, na = dwin[selected_idx]

            return Action(nv=nv, na=na)

        return jax.jit(act)
