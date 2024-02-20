import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import evaluate
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

rouge = evaluate.load('rouge')


@dataclass
class Pair:
    x_l: str
    y_orig: str
    y_summ: str
    compression_ratio: float = None
    density: float = None
    rougeL: float = None
    nli: float = None
    reverse_nli: float = None

    def to_dict(self) -> dict:
        return {
            "x_l": self.x_l,
            "y_orig": self.y_orig,
            "y_summ": self.y_summ,
            "density": float(f"{self.density:.3f}"),
            "rougeL": float(f"{self.rougeL:.3f}"),
            "comp_ratio": float(f"{self.compression_ratio:.3f}"),
            "nli": float(f"{self.nli:.3f}"),
            "reverse_nli": float(f"{self.reverse_nli:.3f}"),
        }


class PairProcessor:
    def __init__(self, args):
        self.device = args.device

        nli_model_name = "alisawuffles/roberta-large-wanli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, use_fast=True)
        self.nli_tokenizer.model_max_length = 512
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.nli_model.eval()

        self.nli_max_batch_size = 2048

    @staticmethod
    def set_comp_ratio(pairs: List[Pair]):
        for pair in pairs:
            pair.compression_ratio = len(pair.y_summ) / len(pair.y_orig)

    @staticmethod
    def set_rouge(pairs: List[Pair]):
        for pair in pairs:
            rougeL_list = rouge.compute(predictions=[pair.y_summ], references=[pair.y_orig],
                                        use_aggregator=False)['rougeL']
            pair.rougeL = rougeL_list[0]

    @staticmethod
    def set_density(pairs: List[Pair]):
        def compute_F(y_orig: str, y_summ: str) -> List[List[str]]:
            orig_token_list, summ_token_list = y_orig.strip().split(), y_summ.strip().split()

            F = []
            i, j = 0, 0

            while i < len(summ_token_list):
                f = []
                while j < len(orig_token_list):
                    if summ_token_list[i] == orig_token_list[j]:
                        i2, j2 = i + 1, j + 1
                        while (j2 < len(orig_token_list) and
                               i2 < len(summ_token_list) and
                               summ_token_list[i2] == orig_token_list[j2]):
                            i2, j2 = i2 + 1, j2 + 1

                        if len(f) < i2 - i:
                            f = summ_token_list[i:i2]
                        j = j2
                    else:
                        j += 1

                i, j = i + max(len(f), 1), 0

                if len(f) > 0:
                    F.append(f)

            return F

        for pair in pairs:
            F = compute_F(pair.y_orig, pair.y_summ)

            density = sum(len(f) ** 2 for f in F) / len(pair.y_summ)
            # coverage = sum(len(f) for f in F) / len(pair.y_summ)

            pair.density = density

    def set_nli(self, pairs: List[Pair]):
        if len(pairs) == 0:
            return

        y_orig_list = [pair.y_orig for pair in pairs]
        y_summ_list = [pair.y_summ for pair in pairs]

        prediction = self.infer_nli(y_orig_list, y_summ_list)
        nli_scores = prediction[:, 1].tolist()

        for pair, nli_score in zip(pairs, nli_scores):
            pair.nli = nli_score

    def set_reverse_nli(self, pairs: List[Pair]):
        if len(pairs) == 0:
            return

        y_orig_list = [pair.y_orig for pair in pairs]
        y_summ_list = [pair.y_summ for pair in pairs]

        prediction = self.infer_nli(y_summ_list, y_orig_list)
        reverse_nli_scores = prediction[:, 1].tolist()

        for pair, reverse_nli_score in zip(pairs, reverse_nli_scores):
            pair.reverse_nli = reverse_nli_score

    def infer_nli(self, premise_list: List[str], hypothesis_list: List[str]) -> torch.LongTensor:
        """
        Infer NLI with given premises and hypotheses. If lists are too long, split and batch-process them.
        :param premise_list: list of premises
        :param hypothesis_list: list of hypotheses
        :return: LongTensor of size (len(premise_list, 3), representing label probabilities
        """

        assert len(premise_list) == len(hypothesis_list), "length of `premise_list` != length of `hypothesis_list`."

        predictions = []
        for start_idx in range(0, len(premise_list), self.nli_max_batch_size):
            batch_premise = premise_list[start_idx:start_idx + self.nli_max_batch_size]
            batch_hypothesis = hypothesis_list[start_idx:start_idx + self.nli_max_batch_size]

            with torch.no_grad():
                input_encoding = self.nli_tokenizer(batch_premise, batch_hypothesis, truncation=True,
                                                    padding=True, return_tensors="pt").to(self.device)

                prediction = F.softmax(self.nli_model(**input_encoding).logits, dim=-1)
                predictions.append(prediction)

        if len(predictions) > 0:
            predictions = torch.cat(predictions, dim=0)  # 0: contradiction, 1: entailment, 2: neutral
        else:
            predictions = torch.LongTensor([])
        return predictions


def save_to_file(sample_list: List[dict], out_fname: Union[Path, str]):
    if len(sample_list) == 0:
        return

    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_fname, "w") as f:
        f.write("\n".join(sample_str_list) + "\n")
