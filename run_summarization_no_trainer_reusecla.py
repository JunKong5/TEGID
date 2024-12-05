#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import sys

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging
import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from filelock import FileLock
from huggingface_hub import Repository
from modeling_bart_kd_reusecla import BartForConditionalGeneration
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,

)
import wandb
import time

devices = torch.device('cuda:0')
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry
from configuration_bart import BartConfig

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

logger = get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.txt"),
        logging.StreamHandler()
    ]
)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--eval_EE", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--test_each_EElayer", action='store_true',
                        help="Set this flag to evaluate each highway.")
    parser.add_argument("--early_exit_entropy", default=-1, type=float,
                        help="Entropy threshold for early exit.")
    parser.add_argument("--output_layer", default=None, type=float,
                        help="Entropy threshold for early exit.")
    parser.add_argument("--loss_type", default=None, type=str,
                        help="loss_type.")
    parser.add_argument("--gamma", default=-1, type=float,
                        help="gamma.")
    parser.add_argument("--temper", default=-1, type=float,
                        help="temper.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,  #############4
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )  ####################
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")  ##############28
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")  ###################
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )  ###############################
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")  ##############
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )  ##############
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )  #################
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )  ##################
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    args = parser.parse_args()
    # args.output_dir = f'./output_{args.output_dir}_{args.dataset_name}/{args.num_train_epochs}_{args.learning_rate}_fp16'

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:  ###########################
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.

    logger.info(accelerator.state, main_process_only=False)
    logger.warning(args)
    logger.warning(args)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir='data')
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = BartConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)
    if args.do_train:
        model.init_early_exiting_eelayer()
    model.model.decoder.set_early_exit_entropy(args.early_exit_entropy)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():  ####可以不用
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
            # num_proc=10,
        )
    # processed_datasets = raw_datasets.map(
    #     preprocess_function,
    #     batched=True,
    #     remove_columns=column_names,
    #     load_from_cache_file=not args.overwrite_cache,
    #     desc="Running tokenizer on dataset",
    #     num_proc=10
    # )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    train_params = []

    for n, p in model.named_parameters():
        if "EElayers" not in n:
            train_params.append((n, p))
        else:
            p.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in train_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states   #####################
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:  ######################
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    # Metric
    metric = evaluate.load("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    prev = 0.0
    exit_layer_counters = {(i + 1): 0 for i in range(config.decoder_layers)}
    total_generated_tokens_len = 0


    if args.do_train:
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(
                        loss_type=args.loss_type,
                        gamma=args.gamma,
                        temper=args.temper,
                        **batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process,
                                                        save_function=accelerator.save)  ###########

                if completed_steps >= args.max_train_steps:
                    completed_steps = 0
                    break
                # for n, p in model.named_parameters():
                #     if "EElayers" in n:
                #         print("---p",p)
            # evaluation
            model.eval()  ########
            # print("args.val_max_target_length",args.val_max_target_length)
            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": args.val_max_target_length if args is not None else config.max_length,
                "num_beams": args.num_beams,
            }

            for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="evaluation"):
                with torch.no_grad():
                    # print("batch[input_ids]",batch["input_ids"].size())
                    generated_tokens, exit_layer_counter = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_layer=args.output_layer,
                        eval_EE=args.eval_EE,
                        **gen_kwargs,
                    )
                    # step = step +1
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    total_generated_tokens_len += len(generated_tokens[0])

                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                                  pad_index=tokenizer.pad_token_id)

                    generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                    generated_tokens = generated_tokens.cpu().numpy()
                    labels = labels.cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds, references=decoded_labels, )
                    for key, value in exit_layer_counter.items():
                        if key in exit_layer_counters:
                            exit_layer_counters[key] += value
                        else:
                            exit_layer_counters[key] = value
                    # if step >20:
                    #     break
            logger.info(f"Exit layer counters = {exit_layer_counters}")
            actual_cost = sum([l * c for l, c in exit_layer_counters.items()])
            full_cost = config.decoder_layers * total_generated_tokens_len
            logger.info(f"  Expected saving = {actual_cost / full_cost}")

            result = metric.compute(use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            logger.info(result)
            res_rougeL = result['rougeLsum']

            if args.with_tracking:
                result["train_loss"] = total_loss.item() / len(train_dataloader)
                result["epoch"] = epoch
                result["step"] = completed_steps
                accelerator.log(result, step=completed_steps)

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process,
                                                save_function=accelerator.save)

            if args.output_dir is not None and res_rougeL > prev:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                prev = res_rougeL
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

    if args.do_test:
        model.to(devices)
        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }
        exit_layer_counters = {(i + 1): 0 for i in range(config.decoder_layers)}
        total_generated_tokens_len = 0
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="evaluation"):
            batch = batch.to(devices)
            with torch.no_grad():

                generated_tokens, exit_layer_counter = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_layer=args.output_layer,
                    eval_EE=args.eval_EE,
                    **gen_kwargs,
                )
                # step = step + 1
                total_generated_tokens_len += len(generated_tokens[0])

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
                for key, value in exit_layer_counter.items():
                    if key in exit_layer_counters:
                        exit_layer_counters[key] += value
                    else:
                        exit_layer_counters[key] = value
                # if step > 20:
                #    break

        logger.info(f"Exit layer counters = {exit_layer_counters}")
        actual_cost = sum([l * c for l, c in exit_layer_counters.items()])
        full_cost = config.decoder_layers * total_generated_tokens_len
        logger.info(f"  Expected saving = {actual_cost / full_cost}")

        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        logger.info(result)

        if args.test_each_EElayer:
            each_layer_results = []
            for i in range(config.decoder_layers):
                model.eval()
                for step, batch in tqdm(enumerate(eval_dataloader), total=20, desc="evaluation"):
                    # bart = BartForConditionalGeneration()
                    # bart.generate()
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            output_layer=i,
                            **gen_kwargs,
                        )
                        step = step + 1
                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not args.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                                      pad_index=tokenizer.pad_token_id)
                        # print(generated_tokens)

                        generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                        generated_tokens = generated_tokens.cpu().numpy()
                        labels = labels.cpu().numpy()

                        if args.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                        # print("decoded_preds", decoded_preds)
                        # print("decoded_labels", decoded_labels)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                        metric.add_batch(
                            predictions=decoded_preds,
                            references=decoded_labels,
                        )
                        if step > 20:
                            break
                result = metric.compute(use_stemmer=True)
                result = {k: round(v * 100, 4) for k, v in result.items()}


if __name__ == "__main__":
    main()
