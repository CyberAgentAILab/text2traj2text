"""Utils for retail environment

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from typing import List, Tuple

import jax
from chex import Array
from jax import numpy as jnp

from ..core.core import Position
from ..core.layout import Layout


@jax.jit
def _compute_pseudo_distance(
    distance_map: Array, start_pos: Array, goal_pos: Array
) -> Array:
    """compute_pseudo_distance

    Args:
        distance_map (Array): 2D array of distance map. each cell is 1 or 4
        start_pos (Array): start position (x,y)
        goal_pos (Array): goal position (x,y)

    Returns:
        Array: pseudo distance
    """
    # Ensure start_pos and goal_pos are integer indices
    start_pos = start_pos.astype(int)
    goal_pos = goal_pos.astype(int)

    # Calculate vertical distance
    v_slice_start = jnp.minimum(start_pos[0], goal_pos[0])
    v_slice_end = jnp.maximum(start_pos[0], goal_pos[0])
    col_index = start_pos[1]

    def v_body_fun(i, total_cost):
        return total_cost + distance_map[i, col_index]

    vertical_move_cost = jax.lax.fori_loop(v_slice_start, v_slice_end, v_body_fun, 0)

    # Calculate horizontal distance
    h_slice_start = jnp.minimum(start_pos[1], goal_pos[1])
    h_slice_end = jnp.maximum(start_pos[1], goal_pos[1])
    row_index = goal_pos[0]

    def h_body_fun(i, total_cost):
        return total_cost + distance_map[row_index, i]

    horizontal_move_cost = jax.lax.fori_loop(h_slice_start, h_slice_end, h_body_fun, 0)

    overlap_cost = distance_map[goal_pos[0], start_pos[1]]

    # Return the sum of vertical and horizontal costs
    return vertical_move_cost + horizontal_move_cost - overlap_cost


def get_nearest_item_index(distance_map: Array, start_pos: Array, goal_pos: Array):
    """get_nearest_item_index

    Args:
        distance_map (Array): 2D array of distance map. each cell is 1 or 4
        start_pos (Array): start position (x,y)
        goal_pos (Array): goal position (x,y)

    Returns:
        Array: nearest item index
    """

    @jax.jit
    def _compute_item_distance(distance_map: Array, start_pos: Array, goal_pos: Array):
        distance = jax.vmap(_compute_pseudo_distance, in_axes=(None, None, 0))(
            distance_map, start_pos, goal_pos
        )
        min_index = jnp.argmin(distance)
        return distance[min_index], min_index

    item_distances, min_index = jax.vmap(
        _compute_item_distance, in_axes=(None, None, 0)
    )(distance_map, start_pos, goal_pos)
    # fori_loop version following

    min_item_index = jnp.argmin(item_distances)
    return min_item_index, min_index[min_item_index]


def get_nearest_item(
    layout: Layout, start_position: Position, item_list: List[str]
) -> Tuple[str, Position]:
    """get_nearest_item

    Args:
        layout (Layout): layout
        start_position (Position): start position
        item_list (List[str]): list of item names

    Returns:
        Tuple[str, Position]: nearest item name and position
    """

    @jax.jit
    def _compute_item_distance(distance_map: Array, start_pos: Array, goal_pos: Array):
        distance = jax.vmap(_compute_pseudo_distance, in_axes=(None, None, 0))(
            distance_map, start_pos, goal_pos
        )
        min_index = jnp.argmin(distance)
        return distance[min_index], min_index

    start_position = start_position.array()
    grid_start_position = start_position * jnp.array(
        [len(layout.item_map[0]), len(layout.item_map)]
    )

    goal_pos = jnp.array(
        [layout.item_to_watching_pos_dict[item_name] for item_name in item_list]
    )
    grid_goal_position = goal_pos * jnp.array(
        [len(layout.item_map[0]), len(layout.item_map)]
    )

    item_distances, min_index = jax.vmap(
        _compute_item_distance, in_axes=(None, None, 0)
    )(jnp.array(layout.pseudo_distance_map), grid_start_position, grid_goal_position)

    min_dist_item_index = jnp.argmin(item_distances)
    item_watching_pos_index = min_index[min_dist_item_index]

    item_to_watch = item_list[min_dist_item_index]
    next_goals = layout.item_to_watching_pos_dict[item_to_watch][
        item_watching_pos_index
    ]
    return item_to_watch, Position.from_array(next_goals)
