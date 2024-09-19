import os
import pathlib
from logging import getLogger

import hydra
import torch
from dotenv import load_dotenv
from text2traj2text.eval.dataset import load_dataset
from text2traj2text.eval.inference import batch_inference
from text2traj2text.eval.utils import compute_metrics, save_results
from text2traj2text.utils import save_config
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

dotenv_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path)


@hydra.main(config_path="config", config_name="eval_llm")
def main(config):

    logger = getLogger(__name__)
    save_config(config, config.logdir)

    if config.quantization.is_quantized:
        compute_dtype = getattr(torch, config.quantization.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.quantization.use_4bit,
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.quantization.use_nested_quant,
        )
    else:
        bnb_config = None
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        use_auth_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=bnb_config,
        use_auth_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
        device_map="auto",  # This will automatically handle device placement
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    test_dataset, human_test_dataset = load_dataset(
        num_examples=config.num_examples,
        bos_token=tokenizer.bos_token,
        seed=config.seed,
    )

    inferences = batch_inference(
        model, tokenizer, test_dataset["text"], config.batch_size
    )
    result_dict = compute_metrics(test_dataset, inferences, is_human_dataset=False)
    save_results(result_dict, logger, f"{config.logdir}/test", is_human_dataset=False)

    inferences = batch_inference(
        model, tokenizer, human_test_dataset["text"], config.batch_size
    )
    result_dict = compute_metrics(human_test_dataset, inferences, is_human_dataset=True)
    save_results(
        result_dict, logger, f"{config.logdir}/human_test", is_human_dataset=True
    )


if __name__ == "__main__":
    main()
