"""Evaluation utilities module.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import csv
import logging
import os
from typing import Any, Dict, List

import nltk
import numpy as np
from datasets import Dataset
from evaluate import load
from text2traj2text.eval.inference import batch_inference
from torch import device
from transformers import PreTrainedModel, PreTrainedTokenizer

MAP_TYPE = ["X", "Y"]
DESCRIPTION_TYPE = ["gpt", "human"]


def evaluate_and_save_results(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    logger: logging.Logger,
    save_dir: str,
    device: device,
    batch_size: int = 16,
    is_human_dataset: bool = False,
) -> None:
    """
    Evaluate model outputs and save results.

    Args:
        model (PreTrainedModel): The model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        dataset (Dataset): The dataset to evaluate on.
        logger (logging.Logger): Logger for output.
        save_dir (str): Directory to save results.
        device (device): Device to run inference on.
        batch_size (int): Batch size for inference.
        is_human_dataset (bool): Whether the dataset is human-generated.
    """
    batch_output = batch_inference(
        model, tokenizer, dataset["text"], device=device, batch_size=batch_size
    )
    result_dict = compute_metrics(dataset, batch_output, is_human_dataset)
    save_results(result_dict, logger, save_dir, is_human_dataset)


def compute_metrics(
    test_dataset: Dataset,
    output_list: List[str],
    is_human_dataset: bool = False,
) -> Dict[str, List[Any]]:
    """
    Compute evaluation metrics for model outputs.

    Args:
        test_dataset (Dataset): The test dataset.
        output_list (List[str]): List of model outputs.
        is_human_dataset (bool): Whether the dataset is human-generated.

    Returns:
        Dict[str, List[Any]]: Dictionary of computed metrics.
    """
    nltk.download("punkt", quiet=True)
    rouge_score = load("rouge")
    bert_score = load("bertscore")

    result_dict = initialize_result_dict(is_human_dataset)

    for i, (text, inference, reference) in enumerate(
        zip(test_dataset["text"], output_list, test_dataset["label"])
    ):
        update_result_dict(
            result_dict,
            text,
            inference,
            reference,
            rouge_score,
            bert_score,
        )
        if is_human_dataset:
            result_dict["description_type"].append(test_dataset["description_type"][i])
            result_dict["map_type"].append(test_dataset["map_type"][i])

    return result_dict


def save_results(
    result_dict: Dict[str, List[Any]],
    logger: logging.Logger,
    save_dir: str,
    is_human_dataset: bool,
) -> None:
    """
    Save evaluation results.

    Args:
        result_dict (Dict[str, List[Any]]): Dictionary of results to save.
        logger (logging.Logger): Logger for output.
        save_dir (str): Directory to save results.
        is_human_dataset (bool): Whether the dataset is human-generated.
    """
    os.makedirs(save_dir, exist_ok=True)

    save_all_results(result_dict, save_dir)
    save_mean_results(result_dict, save_dir)

    logger.info(f"Result: {get_mean_results(result_dict)}")

    if is_human_dataset:
        save_human_dataset_results(result_dict, logger, save_dir)


def save_all_results(result_dict: Dict[str, List[Any]], save_dir: str) -> None:
    """
    Save all individual results to a CSV file.

    Args:
        result_dict (Dict[str, List[Any]]): Dictionary of results to save.
        save_dir (str): Directory to save results.
    """
    with open(f"{save_dir}/all_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result_dict.keys())
        writer.writerows(zip(*result_dict.values()))


def save_mean_results(result_dict: Dict[str, List[Any]], save_dir: str) -> None:
    """
    Save mean results to a CSV file.

    Args:
        result_dict (Dict[str, List[Any]]): Dictionary of results to save.
        save_dir (str): Directory to save results.
    """
    mean_results = get_mean_results(result_dict)
    with open(f"{save_dir}/mean_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(mean_results.keys())
        writer.writerow(mean_results.values())


def get_mean_results(result_dict: Dict[str, List[Any]]) -> Dict[str, float]:
    """
    Calculate mean results for numeric fields.

    Args:
        result_dict (Dict[str, List[Any]]): Dictionary of results.

    Returns:
        Dict[str, float]: Dictionary of mean results.
    """
    return {
        k: np.mean(v)
        for k, v in result_dict.items()
        if k not in ["text", "label", "inference", "description_type", "map_type"]
        and isinstance(v[0], (int, float))
    }


def initialize_result_dict(is_human_dataset: bool) -> Dict[str, List[Any]]:
    """
    Initialize result dictionary based on dataset type.

    Args:
        is_human_dataset (bool): Whether the dataset is human-generated.

    Returns:
        Dict[str, List[Any]]: Initialized result dictionary.
    """
    base_dict = {
        "text": [],
        "label": [],
        "inference": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "rougeLsum": [],
        "bert_f1": [],
        "bert_precision": [],
        "bert_recall": [],
    }
    if is_human_dataset:
        base_dict.update({"description_type": [], "map_type": []})
    return base_dict


def update_result_dict(
    result_dict: Dict[str, List[Any]],
    text: str,
    inference: str,
    reference: str,
    rouge_score: Any,
    bert_score: Any,
) -> None:
    """
    Update result dictionary with computed metrics.

    Args:
        result_dict (Dict[str, List[Any]]): Result dictionary to update.
        text (str): Input text.
        inference (str): Model inference.
        reference (str): Reference text.
        rouge_score (Any): ROUGE score computer.
        bert_score (Any): BERT score computer.
    """
    rouge_result = rouge_score.compute(
        predictions=[inference], references=[reference], use_stemmer=True
    )
    bert_score_result = bert_score.compute(
        predictions=[inference], references=[reference], lang="en"
    )

    result_dict["text"].append(text)
    result_dict["label"].append(reference)
    result_dict["inference"].append(inference)
    for k, v in rouge_result.items():
        result_dict[k].append(v)
    result_dict["bert_precision"].append(np.mean(bert_score_result["precision"]))
    result_dict["bert_recall"].append(np.mean(bert_score_result["recall"]))
    result_dict["bert_f1"].append(np.mean(bert_score_result["f1"]))


def save_human_dataset_results(
    result_dict: Dict[str, List[Any]], logger: logging.Logger, save_dir: str
) -> None:
    """
    Save results for human-generated dataset.

    Args:
        result_dict (Dict[str, List[Any]]): Dictionary of results.
        logger (logging.Logger): Logger for output.
        save_dir (str): Directory to save results.
    """
    os.makedirs(f"{save_dir}/all_results", exist_ok=True)
    save_all_results(result_dict, f"{save_dir}/all_results")
    save_mean_results(result_dict, f"{save_dir}/all_results")

    for desc_type in DESCRIPTION_TYPE:
        logger.info(f"Description type: {desc_type}")
        filtered_results = filter_results(result_dict, "description_type", desc_type)
        os.makedirs(f"{save_dir}/description_type_{desc_type}", exist_ok=True)
        save_all_results(filtered_results, f"{save_dir}/description_type_{desc_type}")
        save_mean_results(filtered_results, f"{save_dir}/description_type_{desc_type}")
        logger.info(f"Mean results: {get_mean_results(filtered_results)}")

    for map_type in MAP_TYPE:
        logger.info(f"Map type: {map_type}")
        filtered_results = filter_results(result_dict, "map_type", map_type)
        os.makedirs(f"{save_dir}/map_type_{map_type}", exist_ok=True)
        save_all_results(filtered_results, f"{save_dir}/map_type_{map_type}")
        save_mean_results(filtered_results, f"{save_dir}/map_type_{map_type}")
        logger.info(f"Mean results: {get_mean_results(filtered_results)}")


def filter_results(
    result_dict: Dict[str, List[Any]], filter_key: str, filter_value: str
) -> Dict[str, List[Any]]:
    """
    Filter result_dict by filter_key and filter_value.

    Args:
        result_dict (Dict[str, List[Any]]): Dictionary to filter.
        filter_key (str): Key to filter on.
        filter_value (str): Value to filter by.

    Returns:
        Dict[str, List[Any]]: Filtered dictionary.
    """
    if filter_key not in result_dict or not isinstance(result_dict[filter_key], list):
        return {}

    indices = [
        i for i, value in enumerate(result_dict[filter_key]) if value == filter_value
    ]

    return {
        k: [v[i] for i in indices]
        if isinstance(v, list) and len(v) == len(result_dict[filter_key])
        else v
        for k, v in result_dict.items()
    }
