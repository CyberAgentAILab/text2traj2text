import csv
import json
import pathlib
from typing import Dict, List

import hydra
from text2traj2text.retail_env.core.layout import Layout
from text2traj2text.retail_env.planner.planner import Planner
from tqdm import tqdm

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"


def load_order(config: Dict) -> Dict:
    with open(
        DATA_DIR
        / "raw_data"
        / "supermarket_env"
        / "floor_plan"
        / f"order_{config.map_type}.json",
        "r",
    ) as f:
        return json.load(f)


def load_floor_map(config: Dict) -> List[List[str]]:
    with open(
        DATA_DIR
        / "raw_data"
        / "supermarket_env"
        / "floor_plan"
        / f"{config.map_type}.csv",
        "r",
    ) as f:
        reader = csv.reader(f)
        return [row for row in reader]


def create_planner(config: Dict, layout: Layout) -> Planner:
    return Planner(
        layout=layout,
        prm_args={"num_samples": config.prm_num_samples},
        dwa_args={"max_steps": config.dwa_max_steps},
        seed=config.seed,
    )


def get_action_plan_files(config: Dict) -> List[pathlib.Path]:
    intent_and_plan_dir = (
        DATA_DIR / "raw_data" / config.project_name / "intent_and_plan"
    )
    return sorted(intent_and_plan_dir.glob("*.json"), key=lambda x: int(x.stem))


def generate_and_save_trajectory(planner: Planner, action_plan_file: pathlib.Path):
    with open(action_plan_file, "r") as f:
        action_plan = json.load(f)

    trajectory = planner.generate_movement_trajectory(
        action_plan["inclined_to_purchase"],
        action_plan["show_interest"],
        action_plan["num_item_to_buy"],
        action_plan["purchase_consideration"],
    )
    output_file_name = action_plan_file.stem + ".json"
    output_file = action_plan_file.parent.parent / "trajectory" / output_file_name
    with open(output_file, "w") as f:
        trajectory_dict = {
            "x": trajectory.pos.x.tolist(),
            "y": trajectory.pos.y.tolist(),
            "r": trajectory.r.tolist(),
            "v": trajectory.v.tolist(),
            "a": trajectory.a.tolist(),
        }
        json.dump(trajectory_dict, f)


@hydra.main(config_path="../config", config_name="dataset_generation")
def main(config):
    order = load_order(config)
    floor_map = load_floor_map(config)

    layout = Layout.from_floor_map_and_item_order(floor_map, order)
    planner = create_planner(config, layout)

    action_plan_files = get_action_plan_files(config)

    (DATA_DIR / "raw_data" / config.project_name / "trajectory").mkdir(
        parents=True, exist_ok=True
    )

    for action_plan_file in tqdm(action_plan_files):
        generate_and_save_trajectory(planner, action_plan_file)


if __name__ == "__main__":
    main()
