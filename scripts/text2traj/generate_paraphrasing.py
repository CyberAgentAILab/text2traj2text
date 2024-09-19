"""generate paraphrased intention from pre-generated intention

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import json
import pathlib

import hydra
from dotenv import load_dotenv
from text2traj2text.text2traj.chain_builder import build_paraphrase_intention_chain
from tqdm import tqdm

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
dot_env_path = BASE_DIR / ".env"

load_dotenv(dot_env_path)


@hydra.main(config_path="../config", config_name="dataset_generation")
def main(config):
    chain = build_paraphrase_intention_chain(config.model_name, config.temperature)

    intent_and_plan_dir = (
        BASE_DIR / "data" / "raw_data" / config.project_name / "intent_and_plan"
    )
    filenames = sorted(intent_and_plan_dir.glob("*.json"), key=lambda x: int(x.stem))

    batch_size = config.batch_size

    intent_list = []
    filename_list = []
    for filename in filenames:
        with open(intent_and_plan_dir / filename, "r") as f:
            intent_and_plan = json.load(f)
        intent_list.append(intent_and_plan["intention"])
        filename_list.append(filename.stem)

    inputs = [
        {"num_paraphrase": config.num_paraphrase, "text": intent}
        for intent in intent_list
    ]

    paraphrase_dir = BASE_DIR / "data" / "raw_data" / config.project_name / "paraphrase"
    paraphrase_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(inputs), batch_size)):
        output_list = chain.batch(inputs[i : i + batch_size])
        temp_filename_list = filename_list[i : i + batch_size]
        for file_name, output in zip(temp_filename_list, output_list):
            temp_dir = paraphrase_dir / file_name
            temp_dir.mkdir(parents=True, exist_ok=True)
            for j, paraphrase in enumerate(
                output["function"].paraphrased_intentions_list
            ):
                intention = paraphrase.paraphrased_intention
                with open(temp_dir / f"{j}.txt", "w") as f:
                    f.write(intention)


if __name__ == "__main__":
    main()
