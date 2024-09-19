"""Retail Store Layout

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from copy import deepcopy
from typing import Dict, List, NamedTuple

import jax
import jax.numpy as jnp
from chex import dataclass
from jax.tree_util import tree_map

from .core import Obstacle, Position

MARGIN = 0.01


@dataclass
class Layout:
    item_map: List[List[str]]
    category_map: List[List[str]]
    category_item_dict: Dict[str, List[str]]
    item_category_dict: Dict[str, str]
    item_to_watching_pos_dict: Dict[str, List[List[float]]]
    adjacent_map: List[List[str]]
    pseudo_distance_map: List[List[int]]
    floor_width: int
    floor_height: int
    obstacles: Obstacle
    casher: Obstacle
    casher_pos: List[List[int]]

    def __post_init__(self):
        self.validate_sample = self._build_validate_sample()
        self.validate_line = self._build_validate_line()

    @classmethod
    def from_floor_map_and_item_order(
        cls, category_map: List[List[str]], item_order: Dict[str, List[str]]
    ) -> "Layout":

        category_map = [
            [cell if cell != "\ufeffwall" else "wall" for cell in row]
            for row in category_map
        ]

        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        floor_height = len(category_map)
        floor_width = len(category_map[0])
        h = 1 / floor_height
        w = 1 / floor_width

        category_list = [category for category in item_order.keys()]

        category_item_dict = deepcopy(item_order)
        item_category_dict = {}
        for category, items in item_order.items():
            for item in items:
                item_category_dict[item] = category

        item_pos_dict = {}
        casher_pos = []

        item_map = deepcopy(category_map)

        for i, row in enumerate(reversed(item_map)):
            for j, cell in enumerate(reversed(row)):
                if cell in category_list:
                    item_list = item_order[cell]
                    if item_list:
                        item_name = item_list.pop(0)
                        row[-1 - j] = item_name  # overwrite floor_map with item name
                        item_pos_dict[item_name] = [
                            len(item_map) - 1 - i,
                            len(row) - 1 - j,
                        ]
                    else:
                        row[-1 - j] = ""  # overwrite cell with empty string
                    for dr, dc in directions:
                        r, c = len(item_map) - 1 - i + dr, len(row) - 1 - j + dc
                        category_map[r][c] = cell
                elif cell == "casher":
                    casher_pos.append([len(item_map) - 1 - i, len(row) - 1 - j])
                # if cell is not in category name list, do nothing

        # pseudo distance map.
        # 1 if cell is empty, 4 if cell is wall or item
        pseudo_distance_map = [
            [1 if cell == "" else 4 for cell in row] for row in item_map
        ]

        # generate item_map and adjacent map to compute item_watching_pos
        adjacent_map = [
            ["" for _ in range(len(item_map[0]))] for _ in range(len(item_map))
        ]

        for name, pos in item_pos_dict.items():
            for dr, dc in directions:
                r, c = pos[0] + dr, pos[1] + dc

                # Check if the adjacent cell is within the map
                adjacent_map[pos[0]][pos[1]] = "wall"
                if (
                    0 <= r < len(item_map)
                    and 0 <= c < len(item_map[0])
                    and item_map[r][c] == ""
                ):
                    adjacent_map[r][c] = name

        # generate item_to_watching_pos_dict
        item_watch_cell_dict = {}
        for i, row in enumerate(adjacent_map):
            for j, item_name in enumerate(row):
                if item_name != "" and item_name != "wall":
                    if item_name not in item_watch_cell_dict.keys():
                        item_watch_cell_dict[item_name] = [[i, j]]
                    else:
                        item_watch_cell_dict[item_name].append([i, j])

        # add padding to align the shape of watching pos
        max_len = max([len(v) for v in item_watch_cell_dict.values()])
        for k, v in item_watch_cell_dict.items():
            # add [1000,1000] as padding
            item_watch_cell_dict[k] = v + [[1000, 1000]] * (max_len - len(v))

        item_watch_pos_dict = {}
        for k, v in item_watch_cell_dict.items():
            item_watch_pos = [[j * w + w / 2, i * h + h / 2] for i, j in v]
            item_watch_pos_dict[k] = item_watch_pos

        obstacles = [
            Obstacle(j * w, i * h, w, h)
            for i in range(floor_height)
            for j in range(floor_width)
            if (item_map[i][j] != "" and item_map[i][j] != "casher")
        ]
        obstacles = tree_map(lambda *values: jnp.stack(values), *obstacles)

        casher = [
            Obstacle(j * w, i * h, w, h)
            for i in range(floor_height)
            for j in range(floor_width)
            if item_map[i][j] == "casher"
        ]
        casher = tree_map(lambda *values: jnp.stack(values), *casher)

        return cls(
            item_map=item_map,
            category_map=category_map,
            category_item_dict=category_item_dict,
            item_category_dict=item_category_dict,
            item_to_watching_pos_dict=item_watch_pos_dict,
            adjacent_map=adjacent_map,
            pseudo_distance_map=pseudo_distance_map,
            floor_width=floor_width,
            floor_height=floor_height,
            obstacles=obstacles,
            casher=casher,
            casher_pos=casher_pos,
        )

    def item_position(self, item_name: str) -> Position:
        position = self.item_to_watching_pos_dict[item_name]
        x = jnp.array([pos[0] for pos in position])
        y = jnp.array([pos[1] for pos in position])
        return Position(x, y)

    def _build_validate_sample(self):
        def validate_sample(sample: Position) -> bool:
            """
            validate if the sample does not collide with obstacles

            Args:
                sample (Position): sample to be validate

            Returns:
                bool: validation result
            """

            def body(sample: Position, obstacle: Obstacle):
                return (
                    (sample.x >= obstacle.x - MARGIN)
                    & (sample.x < obstacle.x + obstacle.w + MARGIN)
                    & (sample.y >= obstacle.y - MARGIN)
                    & (sample.y < obstacle.y + obstacle.h + MARGIN)
                )

            return ~jnp.any(jax.vmap(body, in_axes=(None, 0))(sample, self.obstacles))

        return jax.jit(validate_sample)

    def _build_validate_line(self):
        def validate_line(src: Position, dst: Position) -> bool:
            """
            Validate if the line does not collide with obstacles

            Args:
                src (Position): starting point of line
                dst (Position): end point of line

            Returns:
                bool: validation result
            """

            def body(s: Position, d: Position, obstacle: Obstacle):
                vx = d.x - s.x
                vy = d.y - s.y
                p = jnp.array([-vx, vx, -vy, vy])
                q = jnp.array(
                    [
                        s.x - (obstacle.x - MARGIN),
                        obstacle.x + obstacle.w + MARGIN - s.x,
                        s.y - (obstacle.y - MARGIN),
                        obstacle.y + obstacle.h + MARGIN - s.y,
                    ]
                )

                class Carry(NamedTuple):
                    p: jnp.array
                    q: jnp.array
                    u1: jnp.array
                    u2: jnp.array

                def for_body(i, carry: Carry):
                    t = carry.q[i] / carry.p[i]
                    cond_u1 = (carry.p[i] < 0) & (carry.u1 < t)
                    u1 = t * cond_u1 + carry.u1 * (~cond_u1)
                    cond_u2 = (carry.p[i] > 0) & (carry.u2 > t)
                    u2 = t * cond_u2 + carry.u2 * (~cond_u2)
                    return carry._replace(u1=u1, u2=u2)

                carry = jax.lax.fori_loop(
                    0, 4, for_body, Carry(p=p, q=q, u1=-jnp.inf, u2=jnp.inf)
                )
                u1, u2 = carry.u1, carry.u2

                return ~((u1 > u2) | (u1 > 1) | (u1 < 0) | jnp.any((p == 0) & (q < 0)))

            return ~jnp.any(
                jax.vmap(body, in_axes=(None, None, 0))(src, dst, self.obstacles)
            )

        return jax.jit(validate_line)
