"""Utility functions for the retail environment

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from text2traj2text.retail_env.core.layout import Layout

from .format_trajectory import convert_trajectory_to_textual_description
from .prompt import INPUT_TEMPLATE

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw_data"


def load_sorted_filenames(data_dir: Path) -> List[str]:
    """
    load and sorted filenames
    filename should be like "0.json", "1.json", ...

    Args:
        data_dir (Path): directory of intent_and_plan

    Returns:
        List[str]: sorted filenames
    """
    filenames = [f.name for f in data_dir.iterdir() if f.is_file()]
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))
    return sorted_filenames


def load_dataset_from_intent_filenames(
    base_dataset_dir: Path,
    layout: Layout,
    intent_file_names: List[str],
    stay_duration_threshold: int,
    is_augmented: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Generate augmented dataset from intent filenames.

    Data augmentation is done by paraphrasing original intent.

    Args:
        base_dataset_dir (Path): Base dataset directory.
        layout (Layout): Store layout information.
        intent_file_names (List[str]): Pre-generated intent and plan filenames. like "0.json", "1.json", ...
        stay_duration_threshold (int): Threshold of stay duration.
        is_augmented (bool, optional): Whether the dataset is augmented. Defaults to True.
    Returns:
        Dict[int, Dict[str, Any]]: user activity dict.
    """

    paraphrased_intents = {}
    user_activity_dict = {}

    for intent_filename in intent_file_names:
        (
            traj,
            item_list,
            label,
            paraphrased_intents,
        ) = load_original_and_paraphrased_data(
            base_dataset_dir,
            layout,
            intent_filename,
            stay_duration_threshold,
        )
        intent_index = int(intent_filename.split(".")[0])
        user_activity_text = INPUT_TEMPLATE.format(traj=traj, item_list=item_list)
        user_activity_dict[intent_index] = {
            "text": user_activity_text,
            "label": label,
        }
        if is_augmented:
            user_activity_dict[intent_index][
                "paraphrased_intents"
            ] = paraphrased_intents

    return user_activity_dict


def load_original_and_paraphrased_data(
    base_dataset_dir: Path,
    layout: Layout,
    intent_file_name: str,
    stay_duration_threshold: int,
) -> Tuple[List[str], List[str], str, List[str]]:
    """Load sorted intent and trajectory.

    Args:
        base_dataset_dir (Path): Base dataset directory.
        layout (Layout): Store layout information.
        intent_file_name (str): File name of original intent_and_plan. like "0.json", "1.json", ...
        stay_duration_threshold (int): Threshold of stay duration.
    Returns:
        Tuple[List[str], List[str], str, List[str]]: Lists of trajectory, item list, original intent, and paraphrased intents.
    """
    shopping_plan = load_shopping_plan(base_dataset_dir, intent_file_name)
    item_list = shopping_plan["inclined_to_purchase"]
    original_intent = shopping_plan["intention"]

    intent_index = intent_file_name.split(".")[0]

    raw_trajectory = load_trajectory(base_dataset_dir, intent_index)
    trajectory = convert_trajectory_to_textual_description(
        layout, raw_trajectory, stay_duration_threshold
    )
    paraphrased_intents = get_paraphrased_intents(base_dataset_dir, intent_index)

    return trajectory, item_list, original_intent, paraphrased_intents


def load_shopping_plan(base_dataset_dir: Path, intent_file_name: str) -> Dict:
    with open(base_dataset_dir / "intent_and_plan" / intent_file_name) as f:
        return json.load(f)


def get_paraphrased_intents(
    base_dataset_dir: Path, filename_without_extension: str
) -> List[str]:
    """Get paraphrased intents.

    Args:
        base_dataset_dir (Path): Base dataset directory.
        filename_without_extension (str): File name without extension.

    Returns:
        List[str]: Paraphrased intents.
    """
    paraphrase_dir = base_dataset_dir / "paraphrase" / filename_without_extension
    paraphrase_filenames = load_sorted_filenames(paraphrase_dir)
    return [
        (paraphrase_dir / paraphrase_filename).read_text()
        for paraphrase_filename in paraphrase_filenames
    ]


def load_trajectory(base_dataset_dir: Path, intent_index: str) -> List[List[float]]:
    """Load trajectory from file.

    Args:
        base_dataset_dir (Path): Base dataset directory.
        intent_index (str): Intent index.

    Returns:
        List[List[float]]: Trajectory.
    """
    with open(base_dataset_dir / "trajectory" / f"{intent_index}.json") as f:
        trajectory = json.load(f)
    return [
        [trajectory["x"][i], trajectory["y"][i]] for i in range(len(trajectory["x"]))
    ]


def format_human_data(
    play_log: Dict[str, Any],
    intention: Dict[str, Any],
    human_intention: Dict[str, Any],
    stay_duration_threshold: int,
) -> Tuple[List[str], List[str], str]:
    """Format human data. Generate GPT input and label.

    Args:
        play_log (Dict[str, Any]): Human trajectory data.
        intention (Dict[str, Any]): Intention generated by GPT.
        human_intention (Dict[str, Any]): Human generated intention.
        stay_duration_threshold (int): Threshold of stay duration.

    Returns:
        Tuple[List[str], List[str], str]: Trajectory, item list, and label.
    """

    def get_description(index: int, log_type: str) -> str:
        if log_type == "B" or log_type == "D":
            return human_intention["intentions"][index]["intention"]

        else:
            return intention["intentions"][index]["intention"]

    description_index = play_log["index"]
    label = get_description(description_index, play_log["type"])

    trajectory = play_log["trajectory"]
    map_type = play_log["map"]

    floor_plan_path = DATA_DIR / "supermarket_env" / "floor_plan" / f"{map_type}.csv"
    with floor_plan_path.open("r") as f:
        reader = csv.reader(f)
        floor_map = [row for row in reader]

    item_order_path = (
        DATA_DIR / "supermarket_env" / "floor_plan" / f"order_{map_type}.json"
    )
    with item_order_path.open("r") as f:
        item_order = json.load(f)

    layout = Layout.from_floor_map_and_item_order(floor_map, item_order)
    purchase_item_list = play_log["purchase_item"]

    traj = convert_trajectory_to_textual_description(
        layout,
        trajectory,
        stay_duration_threshold,
    )

    return traj, purchase_item_list, label
