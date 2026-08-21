"""Shared utilities: argument parsing, seed, dataset loading."""

import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from data import SeqRecDataset


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="./configs/TIGER-t5-d128")
    parser.add_argument("--output_dir", type=str, default="./ckpt")
    return parser


def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--index_file", type=str, required=True,
                        help="Index path relative to data/{dataset}/ (e.g. indexes/rkmeans_zcr_cf/Beauty.index.json)")
    parser.add_argument("--max_his_len", type=int, default=20)
    return parser


def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    return parser


def parse_test_args(parser):
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    parser.add_argument("--filter_items", action="store_true", default=True)
    parser.add_argument("--results_file", type=str, default="./results/test.json")
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=None)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--metrics", type=str,
                        default="hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--save_per_user", action="store_true", default=False)
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):
    train_data = ConcatDataset([
        SeqRecDataset(args, mode="train")
    ])
    valid_data = SeqRecDataset(args, "valid")
    return train_data, valid_data


def load_test_dataset(args):
    return SeqRecDataset(args, mode="test", sample_num=args.sample_num)
