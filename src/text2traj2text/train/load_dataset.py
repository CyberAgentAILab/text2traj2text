"""
Load and prepare datasets for training, validation, and testing.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""
import json
from pathlib import Path
from typing import Tuple

from datasets import Dataset
from datasets import load_dataset as huggingface_load_dataset

CURRENT_DIR = Path(__file__).resolve()
DATASET_DIR = CURRENT_DIR.parent.parent.parent.parent / "data"
SEED = 42


def load_datasets(
    validation_size: float = 0.2,
    num_paraphrases: int = 0,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Load train, validation, test, and human test datasets.

    Args:
        validation_size (float): Proportion of data to use for validation.
        num_paraphrases (int): Number of paraphrases to sample for each example.

    Returns:
        Tuple[Dataset, Dataset, Dataset, Dataset]: Train, validation, test, and human test datasets.
    """
    train_validation_dataset = load_augmented_dataset(num_paraphrases, validation_size)
    test_dataset, human_test_dataset = load_test_datasets()

    return (
        train_validation_dataset["train"],
        train_validation_dataset["test"],
        test_dataset,
        human_test_dataset,
    )


def load_test_datasets() -> Tuple[Dataset, Dataset]:
    """
    Load test and human test datasets.

    Returns:
        Tuple[Dataset, Dataset]: Test dataset and human test dataset.
    """
    test_dataset = huggingface_load_dataset(
        "json", data_files=str(DATASET_DIR / "test.json")
    )["train"]
    human_test_dataset = huggingface_load_dataset(
        "json", data_files=str(DATASET_DIR / "human_test.json")
    )["train"]
    return test_dataset, human_test_dataset


def load_augmented_dataset(num_paraphrases: int, validation_size: float) -> Dataset:
    """
    Load and augment the training dataset with paraphrases.

    Args:
        num_paraphrases (int): Number of paraphrases to sample for each example.
        validation_size (float): Proportion of data to use for validation.

    Returns:
        Dataset: Augmented dataset split into train and validation sets.
    """
    with open(DATASET_DIR / "train.json", "r") as f:
        train_dataset_dict = json.load(f)

    texts, labels = [], []
    for item in train_dataset_dict.values():
        texts.append(item["text"])
        labels.append(item["label"])

        if num_paraphrases > 0:
            paraphrased_labels = item["paraphrased_intents"][:num_paraphrases]
            texts.extend([item["text"]] * num_paraphrases)
            labels.extend(paraphrased_labels)

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    return dataset.train_test_split(test_size=validation_size, seed=SEED)
