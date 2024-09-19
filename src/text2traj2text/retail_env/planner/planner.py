"""Planner for the retail environment

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from ..core.core import Position, State
from ..core.layout import Layout
from ..movement_planner.movement_planner import MovementPlanner
from .utils import get_nearest_item

TRAJ = Union[List[int], None]


def normal_distribution(mean: float, std: float, key: PRNGKey):
    return mean + std * jax.random.normal(key, shape=())


class Planner:
    def __init__(
        self,
        layout: Layout,
        traj_planner: MovementPlanner = None,
        prm_args: dict = None,
        dwa_args: dict = None,
        seed: int = 0,
        goal_threshold: float = 0.005,
        num_stop: int = 4,
    ):

        self.key = PRNGKey(seed)
        self.layout = layout
        self.goal_threshold = goal_threshold
        self.num_stop = num_stop
        if traj_planner is None:
            self.traj_planner = MovementPlanner(
                layout=layout, prm_args=prm_args, dwa_args=dwa_args
            )
        else:
            self.traj_planner = traj_planner

    def reset(self, seed):
        self.key = PRNGKey(seed)

    def generate_movement_trajectory(
        self,
        incline_to_purchase: List[str],
        show_interest: List[str],
        num_item_to_buy: int,
        purchase_consideration: int,
        start_pos: List[int] = None,
        category_priority: dict = None,
    ) -> State:

        self.key, subkey = jax.random.split(self.key)

        watch_item_list, category_priority = self._preprocess_for_trajectory_generation(
            subkey,
            incline_to_purchase,
            show_interest,
            num_item_to_buy,
            purchase_consideration,
            category_priority,
        )

        self.key, subkey = jax.random.split(self.key)
        trajectory, _ = self._generate_trajectory(
            subkey, watch_item_list, category_priority, start_pos
        )

        return trajectory

    def update_category_priority(self, key: PRNGKey, purchase_consideration: int):

        category_priority = {
            "household goods": normal_distribution(
                5, 2 + purchase_consideration * 0.2, key
            ),
            "meat": normal_distribution(6, 0.75 + purchase_consideration * 0.2, key),
            "fruit": normal_distribution(10, 0.75 + purchase_consideration * 0.2, key),
            "seasoning": normal_distribution(4, 2 + purchase_consideration * 0.2, key),
            "dairy": normal_distribution(4, 0.75 + purchase_consideration * 0.2, key),
            "seafood": normal_distribution(7, 0.75 + purchase_consideration * 0.2, key),
            "alcohol": normal_distribution(4, 1 + purchase_consideration * 0.2, key),
            "snack": normal_distribution(4, 2 + purchase_consideration * 0.2, key),
            "vegetable": normal_distribution(
                9, 0.75 + purchase_consideration * 0.2, key
            ),
            "drink": normal_distribution(4, 1 + purchase_consideration * 0.2, key),
        }

        # sort category by priority
        category_priority = dict(
            sorted(category_priority.items(), key=lambda x: x[1], reverse=True)
        )
        return category_priority

    def _generate_trajectory(
        self,
        key: PRNGKey,
        item_list: List[str],
        category_priority: dict,
        start_state: State = None,
    ) -> Tuple[List[str], List[Position], List[int]]:

        if start_state is None:
            start_pos = Position(
                1.5 / len(self.layout.item_map),
                (len(self.layout.item_map[0]) - 2) / len(self.layout.item_map[0]),
            )
            start_r = -jnp.pi / 2
        else:
            start_pos = start_state.pos
            start_r = start_state.r

        trajectory = None
        sub_goals = None

        for category in category_priority.keys():
            category_items = self.layout.category_item_dict[category]
            visit_item_list = list(set(category_items) & (set(item_list)))

            while visit_item_list:
                next_goal_item_name, next_goal_pos = get_nearest_item(
                    self.layout, start_pos, visit_item_list
                )

                temp_traj, _ = self.traj_planner.plan(
                    start_pos, start_r, next_goal_pos, self.goal_threshold
                )

                if trajectory is None:
                    trajectory = temp_traj
                    sub_goals = next_goal_pos
                else:
                    trajectory = trajectory.extend(temp_traj)
                    sub_goals = sub_goals.extend(next_goal_pos)

                # add stop phase
                stop_state = State(trajectory.pos.at(-1), trajectory.r[-1], 0, 0)
                for _ in range(self.num_stop):
                    trajectory = trajectory.extend(stop_state)

                start_pos = trajectory.pos.at(-1)
                start_r = trajectory.r[-1]

                # remove already visited item
                visit_item_list.remove(next_goal_item_name)

        # move to casher
        ## pick up casher position randomly
        casher_pos = self._get_casher_position(key)

        temp_traj, _ = self.traj_planner.plan(
            start_pos, start_r, casher_pos, self.goal_threshold
        )

        trajectory = trajectory.extend(temp_traj)
        sub_goals = sub_goals.extend(casher_pos)

        return trajectory, sub_goals

    def _preprocess_for_trajectory_generation(
        self,
        key: PRNGKey,
        incline_to_purchase: List[str],
        show_interest: List[str],
        num_item_to_buy: int,
        purchase_consideration: int,
        category_priority: dict,
    ):

        num_item_to_watch_mean = num_item_to_buy * purchase_consideration / 5

        # generate num item to watch
        key, subkey = jax.random.split(key)
        num_item_to_watch = normal_distribution(
            num_item_to_watch_mean, 3, subkey
        ).astype(int)
        num_item_to_watch = jnp.clip(num_item_to_watch, 0, len(show_interest))

        # sample item to watch
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(
            subkey, len(show_interest), shape=(num_item_to_watch,), replace=False
        )
        show_interest = [show_interest[i] for i in indices]

        # generate category priority
        if category_priority is None:
            key, subkey = jax.random.split(key)
            category_priority = self.update_category_priority(
                subkey, purchase_consideration
            )

        watch_item_list = incline_to_purchase + show_interest
        return watch_item_list, category_priority

    def _get_casher_position(self, key) -> Position:
        casher_index = jax.random.choice(key, len(self.layout.casher_pos))
        casher_x = (self.layout.casher_pos[casher_index][1] + 0.5) / len(
            self.layout.item_map[0]
        )
        casher_y = (self.layout.casher_pos[casher_index][0] + 0.5) / len(
            self.layout.item_map
        )
        return Position(casher_x, casher_y)
