import os
import random
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    args = ArgumentParser(description="T5 Training")
    # args = ArgumentParser(description="BART Training")

    # Files
    args.add_argument("--train_filename", default="./data/processed/train.jsonl", type=str)
    args.add_argument("--valid_filename", default="./data/processed/valid.jsonl", type=str)

    # Experiments
    args.add_argument('--seed', default=999, type=int, help="Random seed.")
    args.add_argument('--save_ckpt', action='store_true', help="Save best checkpoint or not.")
    # args.add_argument('--save_dir', default='/gpfs1/home/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage1/train/gen_model', type=str)
    args.add_argument('--save_dir', default='./gen_model', type=str)
    args.add_argument('--disabled', action='store_true', help="Disable wandb.")

    # Hyperparameters
    args.add_argument('--learning_rate', default=5e-5, type=float)
    args.add_argument('--train_batch_size', default=8, type=int)
    args.add_argument('--valid_batch_size', default=16, type=int)
    args.add_argument('--num_epochs', default=1, type=int)
    args.add_argument('--max_grad_norm', default=1.0, type=float)

    args = args.parse_args()

    # Accelerator
    args.accelerator = Accelerator(log_with="wandb" if not args.disabled else None)

    # CUDA Device
    args.device = args.accelerator.device

    args.accelerator.print(f"Train Filename: {args.train_filename}")
    args.accelerator.print(f"Valid Filename: {args.valid_filename}")

    return args


def init_tokenizer_and_model(args: Namespace) -> Tuple[PreTrainedTokenizerFast, PreTrainedModel]:
    # # For T5
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

    # # For BART
    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
   
    return tokenizer, model




