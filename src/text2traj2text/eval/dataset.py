"""Dataset module for evaluation.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import random
from typing import List, Tuple

from datasets import Dataset
from text2traj2text.train.load_dataset import load_datasets as load_train_datasets

from .prompt import FEW_SHOT_PROMPT, ZERO_SHOT_PROMPT

VALIDATION_SIZE = 0.2


def format_example(train_dataset: Dataset, example_indices: List[int]) -> str:
    """Format examples from the training dataset.

    Args:
        train_dataset (Dataset): Training dataset.
        example_indices (List[int]): Indices of examples to format.

    Returns:
        str: Formatted examples.
    """

    examples = ""
    for idx in example_indices:
        examples += f"{train_dataset[idx]['text']}{train_dataset[idx]['label']}\n\n"
    examples = examples[:-2]
    return examples


def format_dataset(
    train_dataset: Dataset,
    test_dataset: Dataset,
    example_indices: List[List[int]],
    bos_token: str,
    prompt: str,
) -> Dataset:
    """Format the dataset for evaluation.

    Args:
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset to format.
        example_indices (List[List[int]]): List of example indices for each test instance.
        bos_token (str): Beginning of sequence token.
        prompt_template (str): The prompt template to use.

    Returns:
        Dataset: Formatted dataset.
    """

    def format_prompt(i: int) -> str:
        trajectory = test_dataset[i]["text"]
        if example_indices[i]:
            examples = format_example(train_dataset, example_indices[i])
            return prompt.format(
                bos_token=bos_token, trajectory=trajectory, example=examples
            )
        return prompt.format(bos_token=bos_token, trajectory=trajectory)

    text = [format_prompt(i) for i in range(len(test_dataset))]
    return test_dataset.remove_columns(["text"]).add_column("text", text)


def load_dataset(
    num_examples: int,
    bos_token: str = "<bos>",
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load and prepare datasets for evaluation.

    Args:
        num_examples (int): Number of examples for few-shot learning.
        bos_token (str, optional): Beginning of sequence token. Defaults to "<bos>".
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[Dataset, Dataset]: Formatted test dataset and human test dataset.
    """
    random.seed(seed)
    train_dataset, _, test_dataset, human_test_dataset = load_train_datasets()

    def generate_example_indices(dataset_length: int) -> List[List[int]]:
        return [
            random.sample(range(len(train_dataset)), k=num_examples)
            for _ in range(dataset_length)
        ]

    prompt_template = ZERO_SHOT_PROMPT if num_examples == 0 else FEW_SHOT_PROMPT

    test_dataset = format_dataset(
        train_dataset,
        test_dataset,
        generate_example_indices(len(test_dataset)),
        bos_token,
        prompt_template,
    )

    human_test_dataset = format_dataset(
        train_dataset,
        human_test_dataset,
        generate_example_indices(len(human_test_dataset)),
        bos_token,
        prompt_template,
    )

    return test_dataset, human_test_dataset
