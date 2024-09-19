import json
import pathlib

import hydra
from dotenv import load_dotenv
from text2traj2text.text2traj.chain_builder import build_user_intention_chain
from text2traj2text.text2traj.utils import extract_intention

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
dot_env_path = BASE_DIR / ".env"

load_dotenv(dot_env_path)

BATCH_SIZE = 20


@hydra.main(config_path="../config", config_name="dataset_generation.yaml")
def main(config):
    chain = build_user_intention_chain(config.model_name, config.temperature)
    intentions = []

    for i in range(0, config.num_generations, BATCH_SIZE):
        current_batch_size = min(BATCH_SIZE, config.num_generations - i)
        print(f"Generating batch {i // BATCH_SIZE + 1}...")
        current_output = chain.invoke(current_batch_size)
        intentions.extend(extract_intention(current_output))

    # if content is not enough, generate more intentions
    while len(intentions) < config.num_generations:
        num_generations = config.num_generations - len(intentions)
        print(f"Generating {num_generations} more intentions")
        for i in range(0, num_generations, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, num_generations - i)
            current_output = chain.invoke(current_batch_size)
            intentions.extend(extract_intention(current_output))

    save_dir = BASE_DIR / "data" / "raw_data" / config.project_name / "intention"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "intention.json", "w") as f:
        json.dump(intentions, f, indent=4)

    print(f"Generated {len(intentions)} intentions and saved to intention.json")


if __name__ == "__main__":
    main()
