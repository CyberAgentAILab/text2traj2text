import gc
from logging import getLogger

import hydra
import torch
from text2traj2text.eval.utils import evaluate_and_save_results
from text2traj2text.train.load_dataset import load_datasets
from text2traj2text.train.utils import (
    build_compute_metrics_fn,
    build_preprocess_function,
    tokenize_dataset,
)
from text2traj2text.utils import save_config
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)


@hydra.main(config_path="config", config_name="train")
def main(config):

    logger = getLogger(__name__)
    logdir = config.logdir
    save_config(config, logdir)
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(config.train.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        config.train.model_name,
    ).to(device)

    preprocess_function = build_preprocess_function(tokenizer)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        human_test_dataset,
    ) = load_datasets(
        config.dataset.validation_size,
        config.dataset.num_paraphrase,
    )

    tokenized_train_dataset = tokenize_dataset(train_dataset, preprocess_function)
    tokenized_validation_dataset = tokenize_dataset(
        validation_dataset, preprocess_function
    )

    batch_size = config.train.batch_size

    args = Seq2SeqTrainingArguments(
        seed=config.seed,
        logging_dir=f"{config.logdir}/tensorboard",
        output_dir=f"{config.logdir}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.train.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,  # save best and last model
        num_train_epochs=config.train.epoch,
        predict_with_generate=True,
        logging_steps=config.train.logging_steps,
        load_best_model_at_end=True,  # need to load best model checkpoint
        metric_for_best_model="eval_bert_precision",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = build_compute_metrics_fn(tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_model_checkpoint = trainer.state.best_model_checkpoint
    logger.info(f"best_model_checkpoint: {best_model_checkpoint}")

    # # Empty VRAM
    del model
    del trainer
    gc.collect()
    gc.collect()

    model = T5ForConditionalGeneration.from_pretrained(
        best_model_checkpoint,
    ).to(device)

    # evaluation
    logger.info("Evaluating on test dataset:")
    evaluate_and_save_results(
        model,
        tokenizer,
        test_dataset,
        logger,
        f"{logdir}/test",
        device=device,
        batch_size=batch_size,
    )
    logger.info("Evaluating on human test dataset:")
    evaluate_and_save_results(
        model,
        tokenizer,
        human_test_dataset,
        logger,
        f"{logdir}/human_test",
        batch_size=batch_size,
        device=device,
        is_human_dataset=True,
    )


if __name__ == "__main__":
    main()
