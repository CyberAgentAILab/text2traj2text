"""utility functions for training

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from typing import Callable

import nltk
import numpy as np
from datasets import Dataset
from evaluate import load
from nltk.tokenize import sent_tokenize
from transformers.tokenization_utils import PreTrainedTokenizer


def build_compute_metrics_fn(tokenizer: PreTrainedTokenizer) -> Callable:
    """build compute metrics function

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer

    Returns:
        Callable: compute metrics function
    """

    nltk.download("punkt_tab")
    rouge_score = load("rouge")
    bert_score = load("bertscore")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [
            "\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
        ]

        # Compute ROUGE scores
        rouge_result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        # Compute bert score
        bert_score_result = bert_score.compute(
            predictions=decoded_preds, references=decoded_labels, lang="en"
        )
        mean_bert_score = {
            "bert_precision": np.mean(bert_score_result["precision"]),
            "bert_recall": np.mean(bert_score_result["recall"]),
            "bert_f1": np.mean(bert_score_result["f1"]),
        }

        result = {**rouge_result, **mean_bert_score}
        # make all result values negative for early stopping
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def build_preprocess_function(tokenizer: PreTrainedTokenizer) -> callable:
    """build preprocess function for dataset.map

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer

    Returns:
        callable: preprocess function
    """

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["text"], max_length=512, truncation=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["label"], max_length=512, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def tokenize_dataset(
    dataset: Dataset,
    preprocess_function: Callable,
) -> Dataset:
    """tokenize dataset

    Args:
        dataset (Dataset): original dataset
        preprocess_function (Callable): preprocess function for tokenization

    Returns:
        Dataset: tokenized dataset
    """

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)

    return tokenized_dataset
