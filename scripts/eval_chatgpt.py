"""Script for evaluating chatgpt.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import os
import pathlib
from logging import getLogger

import hydra
from dotenv import load_dotenv
from langchain.chat_models.azure_openai import AzureChatOpenAI
from text2traj2text.eval.dataset import load_dataset
from text2traj2text.eval.utils import compute_metrics, save_results
from text2traj2text.utils import save_config

home_dir = pathlib.Path(__file__).parent.parent


@hydra.main(config_path="config", config_name="eval_gpt")
def main(config):
    logger = getLogger(__name__)
    logdir = config.logdir
    save_config(config, logdir)

    load_dotenv(home_dir / ".env")

    test_dataset, human_test_dataset = load_dataset(
        config.num_examples,
        seed=config.seed,
    )

    model = AzureChatOpenAI(
        azure_deployment=config.model.name,
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=config.model.temperature,
    )

    batch_size = config.batch_size

    # call chain.batch per batch_size to avoid exceeding the api call rate limit
    inferences = []
    for i in range(0, len(test_dataset["text"]), batch_size):
        batch_output = model.batch(test_dataset["text"][i : i + batch_size])
        batch_text_output = [output.content for output in batch_output]
        inferences.extend(batch_text_output)
    result_dict = compute_metrics(test_dataset, inferences, is_human_dataset=False)
    save_results(result_dict, logger, f"{logdir}/test", is_human_dataset=False)

    inferences = []
    for i in range(0, len(human_test_dataset["text"]), batch_size):
        batch_output = model.batch(human_test_dataset["text"][i : i + batch_size])
        batch_text_output = [output.content for output in batch_output]
        inferences.extend(batch_text_output)
    result_dict = compute_metrics(human_test_dataset, inferences, is_human_dataset=True)
    save_results(result_dict, logger, f"{logdir}/human_test", is_human_dataset=True)


if __name__ == "__main__":
    main()
