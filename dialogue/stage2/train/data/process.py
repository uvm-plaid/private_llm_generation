import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import evaluate
import ipdb
import jsonlines
from tqdm import tqdm

from util import *


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--device_id", type=int, help="Device to use (0 ~ 7)")
    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.device_id}")

    return args

def split_jsonl_file(input_file_path, train_file_path, valid_file_path, train_percentage=0.8):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Calculate the split index
    split_index = int(len(lines) * train_percentage)

    # Write the train data
    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        for line in lines[:split_index]:
            train_file.write(line)

    # Write the validation data
    with open(valid_file_path, 'w', encoding='utf-8') as valid_file:
        for line in lines[split_index:]:
            valid_file.write(line)


if __name__ == "__main__":
    args = parse_args()
    processor = PairProcessor(args)

    # root_dir = Path("./raw/")
    root_dir = Path("/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage2/generate_and_filter/gen_data/sage-glade-19/")
    # root_dir = Path("/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage2/generate_and_filter/gen_data/vibrant-cherry-10/")
    path = "/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage2/train/data/"
    # root_dir = Path("/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage2/train/data/")

    all_out_pairs = []
    for filename in root_dir.rglob("*.jsonl"):
        out_filename = Path("processed", *Path(filename).parts[1:])

        print(f"Processing {filename} -> {out_filename}")

        with jsonlines.open(filename, "r") as f:
            samples = list(f)

        out_pairs = [Pair(x_l=sample["x_l"], y_orig=sample["y_orig"], y_summ=sample["y_summ"]) for sample in samples]
        with tqdm(total=len(out_pairs)) as pbar:
            for start_idx in range(0, len(out_pairs), args.batch_size):
                batch_pairs = out_pairs[start_idx:start_idx + args.batch_size]

                processor.set_comp_ratio(batch_pairs)
                processor.set_density(batch_pairs)
                processor.set_rouge(batch_pairs)

                processor.set_nli(batch_pairs)
                processor.set_reverse_nli(batch_pairs)

                pbar.update(len(batch_pairs))

        all_out_pairs.extend(out_pairs)

    out_filename = Path(f"{path}/processed/all_out.jsonl")
    print("\nOut Filename: ", out_filename)
    save_to_file([pair.to_dict() for pair in all_out_pairs], out_filename)


    # Split data into input and output
    # input_file_path = 'all.jsonl'
    train_file_path = Path(f"{path}/processed/train.jsonl")
    valid_file_path = Path(f"{path}/processed/valid.jsonl")
    split_jsonl_file(out_filename, train_file_path, valid_file_path, train_percentage=0.8)
