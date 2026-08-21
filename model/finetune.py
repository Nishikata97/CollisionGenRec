"""Training: T5ForConditionalGeneration for sequential recommendation.

Loads T5 config/tokenizer from ckpt/t5-rec/ (d_model=128), adds SID tokens,
and trains with HuggingFace Trainer + early stopping.
"""

import argparse
import os

import torch
import transformers
from transformers import (
    T5Tokenizer, T5Config, T5ForConditionalGeneration,
    EarlyStoppingCallback,
)

from utils import (
    parse_global_args, parse_train_args, parse_dataset_args,
    set_seed, ensure_dir, load_datasets,
)
from collator import Collator


def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    device = torch.device("cuda", local_rank)

    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model, model_max_length=args.model_max_length,
    )

    train_data, valid_data = load_datasets(args)

    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Added {add_num} new tokens.")
        print(f"Train samples: {len(train_data)}")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    # Pre-tokenize
    for ds in train_data.datasets:
        ds.pre_tokenize(tokenizer)
    valid_data.pre_tokenize(tokenizer)
    if local_rank == 0:
        print("Pre-tokenization complete.")

    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if local_rank == 0:
        print(f"Sample train: {train_data[100]}")
        print(f"Sample valid: {valid_data[100]}")

    collator = Collator(tokenizer)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
            dataloader_num_workers=8,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    train(args)
