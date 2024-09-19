"""encode trajectory to description

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from typing import List, Tuple

from text2traj2text.retail_env.core.layout import Layout

STAY_TEMPLATE = "{category}</s>"


def convert_trajectory_to_textual_description(
    layout: Layout,
    traj: List[Tuple[int, int]],
    stay_duration_threshold: int = 5,
) -> str:
    """encode trajectory to description

    Args:
        layout (Layout): store layout information
        traj (List[Tuple[int, int]]): trajectory
        stay_duration_threshold (int, optional): threshold of stay duration. Defaults to 5.

    Returns:
        str: trajectory description
    """

    description = ""

    last_pos = traj[0]

    # for recording stay and move situation
    stay_duration = 0
    last_stay = False

    stop_pos_list = []

    for pos in traj[1:]:
        is_stay = last_pos[0] == pos[0] and last_pos[1] == pos[1]

        # convert continous position to grid world position
        if 0 < pos[0] < 1:
            x = int(pos[0] * layout.floor_width)
            y = int(pos[1] * layout.floor_height - 1)
            current_category = layout.category_map[y][x]
        else:
            current_category = layout.category_map[pos[0]][pos[1]]

        pos = tuple(pos)

        # start to stay.
        if is_stay and not last_stay:
            stay_duration = 1
            last_stay = True
            stay_category = current_category
        # keep staying
        elif is_stay and last_stay:
            stay_duration += 1
            last_stay = True
        # start to move
        elif not is_stay and last_stay:
            if stay_duration >= stay_duration_threshold:
                # recode stay situation
                if stay_category == "":
                    stay_category = "None"
                stay_info = STAY_TEMPLATE.format(
                    category=stay_category,
                )
                stop_pos_list.append(stay_info)

            last_stay = False
            stay_duration = 0
        # keep moving
        elif last_stay:
            last_stay = False
            stay_duration = 0
        last_pos = pos

    for stop_pos in stop_pos_list:
        description += stop_pos

    return description
