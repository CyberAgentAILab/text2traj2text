"""Render trajectory as image

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import matplotlib.pyplot as plt
from text2traj2text.retail_env.core.core import State
from text2traj2text.retail_env.core.layout import Layout

from .utils import ITEM_COLORS, get_position_color


def render(layout: Layout, trajectory: State, save_path: str = "trajectory.pdf"):

    floor_width = len(layout.item_map[0])
    floor_height = len(layout.item_map)

    fig, ax = plt.subplots(figsize=(floor_width / 2, floor_height / 2))

    for i in range(len(layout.obstacles.x)):
        x = int(layout.obstacles.x[i] * floor_width)
        y = int(layout.obstacles.y[i] * floor_height)

        color = get_position_color(x, y, layout)

        ax.add_patch(plt.Rectangle((x, floor_height - y - 1), 1, 1, color=color))

    for i in range(len(layout.casher.x)):
        x = int(layout.casher.x[i] * floor_width)
        y = int(layout.casher.y[i] * floor_height)

        ax.add_patch(
            plt.Rectangle(
                (x, floor_height - y - 1), 1, 1, color=ITEM_COLORS["casher"] / 255
            )
        )

    traj_x = trajectory.pos.x * floor_width
    traj_y = floor_height - trajectory.pos.y * floor_height

    ax.plot(traj_x, traj_y, c="r", linewidth=3)
    ax.scatter(traj_x, traj_y, c="r", s=20)

    ax.set_xlim(0.5, floor_width - 0.5)
    ax.set_ylim(0.5, floor_height - 0.5)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
