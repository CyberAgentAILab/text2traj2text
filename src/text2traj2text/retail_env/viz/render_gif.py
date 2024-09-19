"""Render trajectory as gif

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import io

import matplotlib.pyplot as plt
from PIL import Image
from text2traj2text.retail_env.core.core import State
from text2traj2text.retail_env.core.layout import Layout

from .utils import ITEM_COLORS, get_position_color


def render_gif(
    layout: Layout,
    trajectory: State,
    save_path: str = None,
    file_name: str = "trajectory",
    duration: int = 100,
):
    """render trajectory as gif

    Args:
        layout (Layout): _description_
        trajectory (State): _description_
        save_path (str, optional): _description_. Defaults to None.
        file_name (str, optional): _description_. Defaults to "trajectory".
        duration (int, optional): _description_. Defaults to 100.
    """

    if not save_path:
        save_path = "."

    floor_width = len(layout.item_map[0])
    floor_height = len(layout.item_map)

    fig, ax = plt.subplots(figsize=(floor_width / 2, floor_height / 2))

    # render base image
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

    ax.plot(traj_x, traj_y, c="gray", linewidth=3)
    ax.scatter(traj_x, traj_y, c="gray", s=20)

    ax.set_xlim(0.5, floor_width - 0.5)
    ax.set_ylim(0.5, floor_height - 0.5)
    ax.axis("off")

    # render agent position at each time step
    frames = []
    for i in range(len(traj_x)):
        pos_scatter = ax.scatter(traj_x[i], traj_y[i], c="g", s=250)

        # save each frame to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        frames.append(Image.open(buf))

        # remove scatter
        pos_scatter.remove()

    frames[0].save(
        f"{save_path}/{file_name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def render_partial_view_gif(
    layout: Layout,
    trajectory: State,
    view_size: int = 5,
    save_path: str = None,
    file_name: str = "trajectory",
    duration: int = 100,
):
    """render trajectory as gif

    Args:
        layout (Layout): _description_
        trajectory (State): _description_
        view_size (int, optional): _description_. Defaults to 5.
        save_path (str, optional): _description_. Defaults to None.
        file_name (str, optional): _description_. Defaults to "trajectory".
        duration (int, optional): _description_. Defaults to 100.
    """

    if not save_path:
        save_path = "."

    floor_width = len(layout.item_map[0])
    floor_height = len(layout.item_map)

    fig, ax = plt.subplots(figsize=(10, 10))

    # render base image
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

    ax.plot(traj_x, traj_y, c="gray", linewidth=3)
    ax.scatter(traj_x, traj_y, c="gray", s=20)

    ax.axis("off")
    ax.spines["top"].set_linewidth(5)
    ax.spines["right"].set_linewidth(5)
    ax.spines["left"].set_linewidth(5)
    ax.spines["bottom"].set_linewidth(5)

    # render agent position at each time step
    frames = []
    for i in range(len(traj_x)):
        pos_scatter = ax.scatter(traj_x[i], traj_y[i], c="g", s=500)

        ax.set_xlim(traj_x[i] - view_size, traj_x[i] + view_size)
        ax.set_ylim(traj_y[i] - view_size, traj_y[i] + view_size)

        # save each frame to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        frames.append(Image.open(buf))

        # remove scatter
        pos_scatter.remove()

    frames[0].save(
        f"{save_path}/{file_name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
