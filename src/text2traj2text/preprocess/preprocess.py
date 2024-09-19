"""
make train/validation/test dataset from pre-generated trajectory and paraphrase

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""
import csv
import json
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
from datasets import Dataset
from text2traj2text.preprocess.prompt import INPUT_TEMPLATE
from text2traj2text.preprocess.utils import (
    format_human_data,
    load_dataset_from_intent_filenames,
    load_sorted_filenames,
)
from text2traj2text.retail_env.core.layout import Layout

current_path = Path(__file__).resolve()
BASE_DIR = current_path.parent.parent.parent.parent / "data"
DATA_DIR = BASE_DIR / "raw_data"


def format_synthesized_dataset(
    dataset_name: str,
    test_size: float,
    seed: int,
    stay_duration_threshold: int,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """Format dataset from pre-generated user activity dataset

    Args:
        dataset_name (str): dataset name
        test_size (float): dataset split rate for train_validation/test
    """
    base_dataset_dir = DATA_DIR / dataset_name
    intent_file_names = load_sorted_filenames(base_dataset_dir / "intent_and_plan")
    np.random.seed(seed)
    np.random.shuffle(intent_file_names)

    train_intent_filenames = intent_file_names[
        : int(len(intent_file_names) * (1 - test_size))
    ]
    test_intent_filenames = intent_file_names[
        int(len(intent_file_names) * (1 - test_size)) :
    ]

    with open(DATA_DIR / "supermarket_env" / "floor_plan" / "order_X.json", "r") as f:
        order = json.load(f)
    with open(DATA_DIR / "supermarket_env" / "floor_plan" / "X.csv", "r") as f:
        reader = csv.reader(f)
        floor_map = [row for row in reader]
    layout = Layout.from_floor_map_and_item_order(floor_map, order)

    train_user_activity_dict = load_dataset_from_intent_filenames(
        base_dataset_dir,
        layout,
        train_intent_filenames,
        stay_duration_threshold,
        is_augmented=True,
    )

    with open(BASE_DIR / "train.json", "w") as f:
        json.dump(train_user_activity_dict, f)

    # don't augment test dataset
    test_user_activity_dict = load_dataset_from_intent_filenames(
        base_dataset_dir,
        layout,
        test_intent_filenames,
        stay_duration_threshold,
        is_augmented=False,
    )
    test_user_activity_dict = {
        "text": [activity["text"] for activity in test_user_activity_dict.values()],
        "label": [activity["label"] for activity in test_user_activity_dict.values()],
    }
    dataset = Dataset.from_dict(test_user_activity_dict)
    dataset.to_json(str(BASE_DIR / "test.json"))


def format_human_test_data(
    human_stay_duration_threshold: int,
) -> None:
    """Format human test dataset

    Args:
        human_stay_duration_threshold (int): threshold of stay duration for human test dataset
    """

    synthesized_intention_path = DATA_DIR / "text2traj" / "intention" / "intention.json"
    with open(synthesized_intention_path, "r") as f:
        intention = json.load(f)

    synthesized_human_intention_path = (
        DATA_DIR / "text2traj" / "intention" / "human_intention.json"
    )
    with open(synthesized_human_intention_path, "r") as f:
        human_intention = json.load(f)

    dataset_dict = {"text": [], "label": [], "description_type": [], "map_type": []}

    # load as pathlib.Path
    test_data_dir_list = [
        d
        for d in Path(DATA_DIR / "human_test_data").iterdir()
        if d.name.startswith("participant")
    ]
    for test_data_dir in test_data_dir_list:
        file_name_list = [
            f for f in Path(test_data_dir).iterdir() if f.suffix == ".json"
        ]
        for filename in file_name_list:
            with open(filename, "r") as f:
                play_log = json.load(f)
            traj, item_list, label = format_human_data(
                play_log, intention, human_intention, human_stay_duration_threshold
            )
            text = INPUT_TEMPLATE.format(traj=traj, item_list=item_list)
            description_type = "gpt" if play_log["type"] in ["A", "C"] else "human"
            dataset_dict["text"].append(text)
            dataset_dict["label"].append(label)
            dataset_dict["description_type"].append(description_type)
            dataset_dict["map_type"].append(play_log["map"])

    dataset = Dataset.from_dict(dataset_dict)
    dataset.to_json(str(BASE_DIR / "human_test.json"))


@hydra.main(config_path="config", config_name="default")
def main(config):
    format_synthesized_dataset(
        config.dataset_name,
        config.test_size,
        config.seed,
        config.stay_duration_threshold,
    )
    format_human_test_data(config.human_stay_duration_threshold)


if __name__ == "__main__":
    main()
