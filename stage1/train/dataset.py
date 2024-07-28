import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Dict
from collections import Counter

import ipdb
import jsonlines
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer


@dataclass
class CTGExample:
    x_l: str
    y_orig: str
    y_summ: str
    comp_ratio: float
    density: float
    rougeL: float
    nli: float
    reverse_nli: float
    length_type: str
    abstract_type: str

    def __init__(self, x_l, y_orig, y_summ, comp_ratio, density, rougeL, nli, reverse_nli):
        self.x_l = x_l
        self.y_orig = y_orig
        self.y_summ = y_summ
        self.comp_ratio = comp_ratio
        self.density = density
        self.rougeL = rougeL
        self.nli = nli
        self.reverse_nli = reverse_nli

        if comp_ratio < 0.5:
            self.length_type = "summary-short"
        elif 0.5 <= comp_ratio < 0.8:
            self.length_type = "summary-long"
        else:
            self.length_type = "paraphrase"

        self.abstract_type = "abstractive" if max([density, rougeL]) < 0.6 else "extractive"

    @staticmethod
    def from_json(json_data: Dict):
        return CTGExample(
            x_l=json_data["x_l"],
            y_orig=json_data["y_orig"],
            y_summ=json_data["y_summ"],
            comp_ratio=json_data["comp_ratio"],
            density=json_data["density"],
            rougeL=json_data["rougeL"],
            nli=json_data["nli"],
            reverse_nli=json_data["reverse_nli"],
        )


class CTGDataset(Dataset):
    tokenizer: PreTrainedTokenizerFast = None

    def __init__(self, tokenizer: PreTrainedTokenizerFast, example_list: List[CTGExample]):
        CTGDataset.tokenizer = tokenizer
        self.example_list = example_list

    def __len__(self):
        return len(self.example_list)

    def __iter__(self):
        return iter(self.example_list)

    def __getitem__(self, idx):
        return self.example_list[idx]

    @staticmethod
    def format_prompt(example: CTGExample) -> str:
        if example.length_type == "summary-short":
            prompt = f"Generate a short and {example.abstract_type} summary of the given sentence.\n" \
                     f"Sentence: {example.y_orig}"

        elif example.length_type == "summary-long":
            prompt = f"Generate a long and {example.abstract_type} summary of the given sentence.\n" \
                     f"Sentence: {example.y_orig}"

        else:
            prompt = f"Generate a paraphrase of the given sentence.\n" \
                     f"Sentence: {example.y_orig}"

        return prompt

    @staticmethod
    def from_file(tokenizer: PreTrainedTokenizerFast, filename: str):
        with jsonlines.open(filename, 'r') as f:
            example_list = [CTGExample.from_json(data) for data in f]

        return CTGDataset(tokenizer, example_list)

    @staticmethod
    def collate_fn(batched_examples: List[CTGExample]) -> Namespace:
        batched_prompts = [CTGDataset.format_prompt(example)
                           for example in batched_examples]
        batched_y_summs = [example.y_summ for example in batched_examples]

        prompt_encoding = CTGDataset.tokenizer(batched_prompts, padding=True,
                                               truncation=True, return_tensors="pt")
        y_summ_encoding = CTGDataset.tokenizer(batched_y_summs, padding=True,
                                               truncation=True, return_tensors="pt")

        return Namespace(
            prompt_encoding=prompt_encoding,
            y_summ_encoding=y_summ_encoding,
        )

