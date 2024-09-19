import json
import pathlib

import hydra
from dotenv import load_dotenv
from text2traj2text.text2traj.chain_builder import build_preferred_item_chain

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
dot_env_path = BASE_DIR / ".env"

load_dotenv(dot_env_path)


@hydra.main(config_path="../config", config_name="dataset_generation")
def main(config):

    chain = build_preferred_item_chain(config.model_name, config.temperature)

    with open(
        BASE_DIR
        / "data"
        / "raw_data"
        / config.project_name
        / "intention"
        / "intention.json",
        "r",
    ) as f:
        intentions = json.load(f)

    save_dir = BASE_DIR / "data" / "raw_data" / config.project_name / "intent_and_plan"
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, intention in enumerate(intentions):
        input = {
            "num_item": intention["num_item_to_buy"],
            "intention": intention["intention"],
            "purchase_consideration": intention["purchase_consideration"],
        }
        model_output = chain.invoke(input)

        shopping_plan = model_output["output"]["shopping_plan"]
        preferred_item_list = model_output["output"]["preffer_item"]

        result = {
            "intention": input["intention"],
            "num_item_to_buy": input["num_item"],
            "purchase_consideration": int(input["purchase_consideration"]),
            "shopping_plan": {
                plan.category: plan.num_purchase_item for plan in shopping_plan
            },
            "inclined_to_purchase": preferred_item_list["incline_to_purchase"],
            "show_interest": preferred_item_list["show_interest"],
        }

        with open(
            save_dir / f"{i}.json",
            "w",
        ) as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
