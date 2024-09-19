"""Inference module for evaluation.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from typing import List

from datasets import Dataset
from torch import device
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)


def batch_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    device: device,
    batch_size: int = 16,
) -> List[str]:
    """batch inference

    Args:
        model (PreTrainedModel): pre-trained model
        tokenizer (PreTrainedTokenizer): tokenizer
        dataset (Dataset): dataset for inference
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        List[str]: list of generated texts
    """
    if isinstance(model, T5ForConditionalGeneration):
        pipe = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            device=device,
        )
        text_key = "summary_text"
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            return_full_text=False,
            max_new_tokens=200,
            device=device,
        )
        text_key = "generated_text"

    output_list = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="inference"):
        batch_output = pipe(dataset[i : i + batch_size])
        if isinstance(batch_output[0], list):
            batch_output = [output[0][text_key] for output in batch_output]
        else:
            batch_output = [output[text_key] for output in batch_output]
        output_list.extend(batch_output)
    return output_list
