import os
import re
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Iterable, Union

import ipdb
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig, \
    AutoModelForSeq2SeqLM

from pipeline.util import Candidates, bio_gpt_postprocess


class ConGenerator:
    def __init__(self, args: Namespace):
        self.args = args

        self.tokenizer, self.model = self.init_tokenizer_and_model()
        self.generation_config = self.init_generation_config()

        self.spacy_model = spacy.load("en_core_web_sm")

    def init_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        tokenizer = AutoTokenizer.from_pretrained("t5-large")

        model_name = self.args.model_name
        print("Model name: ", model_name)
        model_path = f"/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage_dp/model/{model_name}"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.args.device)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.args.device)

        return tokenizer, model

    def init_generation_config(self) -> GenerationConfig:
        generation_config = GenerationConfig(
            max_new_tokens=100, num_return_sequences=10,
            do_sample=True, top_p=0.9, temperature=0.7, no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return generation_config

    def generate_y_cons_from_sample(self, samples: List[dict]) -> List[Candidates]:
        """
        Batched version of the full pipeline generating `y_con`s
        :param samples: list of samples with "x_l", "y_orig"
        :return: out_sample with "x_l", "y_orig" and "y_cons"
        """
        if self.args.domain == "reddit":
            for sample in samples:
                if "Text: " in sample["x_l"][:30]:
                    prefix_end_idx = sample["x_l"].find("Text: ") + len("Text: ")
                    sample["x_l"] = sample["x_l"][prefix_end_idx:]

        prefixes = [prefix for sample in samples for prefix in self.format_prompt(sample)]

        generations = self.generate_with_prefix(prefixes)
        num_gen_per_sample = len(generations) // len(samples)

        candidates = []
        for idx, (prefix, sample) in enumerate(zip(prefixes, samples)):
            generations_for_this_prefix = generations[num_gen_per_sample * idx:num_gen_per_sample * (idx + 1)]
            y_cons_for_this_prefix = [self.postprocess_generation(text) for text in generations_for_this_prefix]
            y_cons_for_this_prefix = [y_con for y_con in y_cons_for_this_prefix if self.qualifies_as_y_con(y_con)]

            candidates.append(
                Candidates(x_l=sample["x_l"], y_orig=sample["y_orig"], y_cons=y_cons_for_this_prefix)
            )

        return candidates

    def generate_with_prefix(self, prefix: Union[str, List[str]]) -> List[str]:
        input_encoding = self.tokenizer(prefix, return_tensors="pt", truncation=True, padding=True).to(self.args.device)

        outputs = self.model.generate(
            **input_encoding,
            generation_config=self.generation_config,
        )

        outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs_str

    @staticmethod
    def format_prompt(example: dict) -> List[str]:
        prompt_list = [
            f"Generate a short and abstractive summary of the given sentence.\n"
            f"Sentence: {example['y_orig']}",
            f"Generate a short and extractive summary of the given sentence.\n"
            f"Sentence: {example['y_orig']}",
            f"Generate a long and abstractive summary of the given sentence.\n"
            f"Sentence: {example['y_orig']}",
            f"Generate a long and extractive summary of the given sentence.\n"
            f"Sentence: {example['y_orig']}",
            f"Generate a paraphrase of the given sentence.\n"
            f"Sentence: {example['y_orig']}",
        ]

        return prompt_list

    def postprocess_generation(self, text: str) -> str:
        postprocessed = text.strip()
        return postprocessed

    def split_sentences(self, text: str) -> List[str]:
        return [str(sent).strip() for sent in self.spacy_model(text).sents]

    def qualifies_as_y_con(self, text: str) -> bool:
        """Given text, determine whether text qualifies as a legit y_con"""
        if self.args.domain in ["news", "news-human"]:
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        elif self.args.domain in ["reddit", "reddit-human"]:
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            no_link = "http" not in text
            no_edit = len(re.findall(r'edit([\d\s]+)?:', text.lower())) == 0
            out = default and no_link and no_edit

        elif self.args.domain in ["bio", "bio-human"]:
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        else:
            raise NotImplementedError

        return out

