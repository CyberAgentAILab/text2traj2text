"""Base class for local planner

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from numpy.typing import NDArray

from ..core.core import Position, State
from ..core.layout import Layout


class Action(NamedTuple):
    nv: float
    na: float


class Obstacle(NamedTuple):
    x: float
    y: float
    w: float
    h: float


@dataclass
class BaseLocalPlanner:
    layout: Layout
    max_vel: float = 0.05
    max_acc: float = 0.01
    max_ang: float = 0.25 * jnp.pi
    max_ang_acc: float = 0.125 * jnp.pi
    max_steps: int = 100

    def __post_init__(self):
        self.compute_next_state = self._build_compute_next_state()
        self.act = self._build_act()
        self._plan = self._build_plan()

    def _build_compute_next_state(self):
        def compute_next_state(state: State, action: Action) -> State:
            """
            Compute next state from the current state and selected action

            Args:
                state (State): current state
                action (Action): seleted action

            Returns:
                State: next state
            """
            vel = action.nv * self.max_vel
            ang = action.na * self.max_ang
            next_rot = (state.r + ang) % (2 * jnp.pi)
            next_x = state.pos.x + vel * jnp.cos(next_rot)
            next_y = state.pos.y + vel * jnp.sin(next_rot)
            validity = self.layout.validate_line(state.pos, Position(next_x, next_y))
            next_x = next_x * validity + state.pos.x * ~validity
            next_y = next_y * validity + state.pos.y * ~validity
            next_pos = Position(x=next_x, y=next_y)
            vel = vel * validity
            ang = ang * validity

            return State(pos=next_pos, r=next_rot, v=vel, a=ang)

        return jax.jit(compute_next_state)

    def _build_act(self):
        def act(state: State, goal_pos: Array) -> Action:
            """
            Select an action

            Args:
                state (State): current state
                goal_pos (Array): goal position

            Returns:
                Action: selected action
            """
            raise NotImplementedError()

        return jax.jit(act)

    def _build_plan(self):
        def _plan(
            start_state: State,
            goal_pos: Position,
            goal_radius: float,
        ) -> Position:
            """
            Plan a trajectory

            Args:
                start_state (State): Start state
                goal_pos (Position): goal position
                goal_radius (float): the radius of goal region

            Returns:
                Position: Planned trajectory

            Note:
                there may be a bug around the inconsistent type of states before/after the body function

            """

            class Carry(NamedTuple):
                t: int
                traj: Array
                state: State

            def cond(carry: Carry) -> bool:
                goal = (
                    jnp.linalg.norm((carry.state.pos - goal_pos).array()) > goal_radius
                )
                time = carry.t < self.max_steps

                return goal & time

            def update_traj(traj, state: State, t):
                return traj.at[t].set(
                    jnp.hstack((state.pos.array(), state.r, state.v, state.a))
                )

            def body(carry: Carry) -> Carry:
                traj = update_traj(carry.traj, carry.state, carry.t)
                action = self.act(carry.state, goal_pos)
                state = self.compute_next_state(carry.state, action)

                return Carry(traj=traj, state=state, t=carry.t + 1)

            state = start_state
            traj = jnp.ones((self.max_steps, 5)) * jnp.inf
            carry = Carry(traj=traj, state=state, t=0)
            carry = jax.lax.while_loop(cond, body, carry)
            traj = update_traj(carry.traj, carry.state, carry.t)

            return traj

        return jax.jit(_plan)

    def plan(
        self,
        start_state: State,
        goal_pos: Position,
        goal_radius: float,
    ) -> NDArray:
        """
        Plan a trajectory

        Args:
            start_state (State): Start state
            goal_pos (Position): goal position
            goal_radius (float): the radius of goal region

        Returns:
            NDArray: Planned trajectory in numpy array
        """
        raw_traj = jnp.asarray(self._plan(start_state, goal_pos, goal_radius))
        traj = raw_traj[~jnp.isinf(raw_traj[:, 0])]

        return traj
