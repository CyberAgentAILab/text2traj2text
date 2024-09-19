"""Core module for retail environment.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import os
from typing import Dict, NamedTuple

import jax.numpy as jnp
import numpy as np
from chex import Array
from numpy.typing import NDArray

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)

ITEM_COLORS = {
    "": [255, 255, 255],
    "wall": [100, 100, 100],
    "item": [150, 150, 150],
    "casher": [125, 125, 125],
}


class Position(NamedTuple):
    x: float
    y: float

    def array(self) -> Array:
        if jnp.array(self.x).ndim < 1:
            return jnp.array([self.x, self.y])
        else:
            return jnp.column_stack([self.x, self.y])

    @classmethod
    def from_array(cls, arr: NDArray) -> "Position":
        return cls(arr[0], arr[1])

    def at(self, index: int) -> "Position":
        return Position(self.x[index], self.y[index])

    def extend(self, other_pos: "Position") -> "Position":
        return Position(
            jnp.hstack([self.x, other_pos.x]), jnp.hstack([self.y, other_pos.y])
        )

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)

    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self):
        return iter(Position(self.x[i], self.y[i]) for i in range(len(self.x)))


class Item:
    def __init__(
        self, x: int, y: int, item_name: Dict[str, str], item_info: str = None
    ):
        """initialize item

        Args:
            x (int): item x position
            y (int): item y position
            item_name (str): item name
            item_info (Dict[str, str], optional): item information. Defaults to None.
        """

        self.x = x
        self.y = y
        self.name = item_name
        self.display_name = item_name.split(" ")
        self.item_info = item_info

    def get_info(self) -> str:
        """get item information

        Returns:
            str: item information
        """
        info = ""

        if self.item_info is None:
            return ""

        for key, value in self.item_info.items():
            if key == "name":
                info += f"{value}\n"
            else:
                info += f"   {key}: {value}\n"
        return info


class State(NamedTuple):
    pos: Position
    r: float
    v: float
    a: float

    def at(self, index: int) -> "State":
        return State(self.pos.at(index), self.r[index], self.v[index], self.a[index])

    def extend(self, other_state: "State") -> "State":
        return State(
            self.pos.extend(other_state.pos),
            jnp.hstack([self.r, other_state.r]),
            jnp.hstack([self.v, other_state.v]),
            jnp.hstack([self.a, other_state.a]),
        )

    def to_dict(self) -> Dict[str, NDArray]:
        return {
            "x": self.pos.x.tolist(),
            "y": self.pos.y.tolist(),
            "r": self.r.tolist(),
            "v": self.v.tolist(),
            "a": self.a.tolist(),
        }

    def __iter__(self):
        return iter(
            State(
                self.pos.at(i),
                self.r[i],
                self.v[i],
                self.a[i],
            )
            for i in range(len(self.pos))
        )


class Action(NamedTuple):
    nv: float
    na: float


class Obstacle(NamedTuple):
    x: float
    y: float
    w: float
    h: float
