import math
from argparse import Namespace

import ipdb
import torch
import torch.nn as nn
import wandb as wandb
from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from accelerate import Accelerator
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizerFast

from dataset import *
from util import *


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                    args: Namespace, epoch: int):
    model.train()

    num_oom_batch = 0

    with tqdm(dataloader, desc=f"Train Ep {epoch}", total=len(dataloader),
              disable=not args.accelerator.is_local_main_process) as tq:
        for batch in tq:
            prompt_encoding = batch.prompt_encoding
            y_summ_encoding = batch.y_summ_encoding

            labels = y_summ_encoding.input_ids.masked_fill(
                y_summ_encoding.input_ids == args.accelerator.unwrap_model(model).config.pad_token_id, -100
            )

            try:
                output = model(**prompt_encoding, labels=labels)
                loss = output.loss
                args.accelerator.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                ppl = math.exp(loss.detach().item())

                args.accelerator.log({"train_PPL": ppl})
                tq.set_postfix({"OOM": num_oom_batch})

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad()

                num_oom_batch += 1

    args.accelerator.print(f"Number of OOM batches: {num_oom_batch}")


# def evaluate(model: nn.Module, dataloader: DataLoader, tokenizer: PreTrainedTokenizerFast,
#              args: Namespace, epoch: int, wandb_table: wandb.Table):
#     model.eval()

#     generation_batch = None
#     ppl_sum = 0.
#     total_sample_count = 0.
#     with tqdm(dataloader, desc=f"Eval", total=len(dataloader),
#               disable=not args.accelerator.is_local_main_process) as tq:
#         for batch in tq:
#             prompt_encoding = batch.prompt_encoding
#             y_summ_encoding = batch.y_summ_encoding

#             labels = y_summ_encoding.input_ids.masked_fill(
#                 y_summ_encoding.input_ids == args.accelerator.unwrap_model(model).config.pad_token_id, -100
#             )

#             with torch.no_grad():
#                 output = model(**prompt_encoding, labels=labels)
#                 loss = output.loss

#                 batch_sample_count = labels.size(0)
#                 ppl_sum += math.exp(loss.item()) * batch_sample_count
#                 total_sample_count += batch_sample_count

#             if generation_batch is None:
#                 generation_batch = batch

#     avg_ppl = ppl_sum / total_sample_count
#     args.accelerator.log({"eval_avg_ppl": avg_ppl})

#     # Generate with beam search and log original input, label, and prediction
#     generation_table = None
#     if args.accelerator.is_main_process:
#         generation_config = GenerationConfig(
#             max_new_tokens=50, num_return_sequences=1,
#             pad_token_id=args.accelerator.unwrap_model(model).config.pad_token_id,
#             do_sample=True, top_p=0.9, temperature=0.5
#         )
#         prompt_encoding = generation_batch.prompt_encoding.to(args.device)
#         output_sequences = tokenizer.batch_decode(args.accelerator.unwrap_model(model).generate(
#             **prompt_encoding,
#             **vars(generation_config)  # Ensure the generation_config is correctly unpacked
#         ), skip_special_tokens=True)

#         # Decode original texts and labels
#         original_texts = tokenizer.batch_decode(generation_batch.prompt_encoding.input_ids, skip_special_tokens=True)
#         original_labels = tokenizer.batch_decode(generation_batch.y_summ_encoding.input_ids, skip_special_tokens=True)

#         if wandb_table is not None:
#             for original_text, original_label, predicted_text in zip(original_texts, original_labels, output_sequences):
#                 # Adjust this line to match the column names of your wandb.Table
#                 wandb_table.add_data(epoch, original_text, original_label, predicted_text)
#             generation_table = wandb.Table(columns=wandb_table.columns, data=wandb_table.data)
#             args.accelerator.log({"Generation": generation_table})

#     return avg_ppl, generation_table


def evaluate(model: nn.Module, dataloader: DataLoader, tokenizer: PreTrainedTokenizerFast,
             args: Namespace, epoch: int, wandb_table: wandb.Table):
    model.eval()

    generation_batch = None
    ppl_sum = 0.
    total_sample_count = 0.
    with tqdm(dataloader, desc=f"Eval", total=len(dataloader),
              disable=not args.accelerator.is_local_main_process) as tq:
        for batch in tq:
            prompt_encoding = batch.prompt_encoding
            y_summ_encoding = batch.y_summ_encoding

            labels = y_summ_encoding.input_ids.masked_fill(
                y_summ_encoding.input_ids == args.accelerator.unwrap_model(model).config.pad_token_id, -100
            )

            with torch.no_grad():
                output = model(**prompt_encoding, labels=labels)
                loss = output.loss

                batch_sample_count = labels.size(0)
                ppl_sum += math.exp(loss.item()) * batch_sample_count
                total_sample_count += batch_sample_count

            if generation_batch is None:
                generation_batch = batch

    avg_ppl = ppl_sum / total_sample_count
    args.accelerator.log({"eval_avg_ppl": avg_ppl})

    # Generate with beam search
    generation_table = None
    if args.accelerator.is_main_process:
        generation_config = GenerationConfig(
            max_new_tokens=50, num_return_sequences=1,
            pad_token_id=args.accelerator.unwrap_model(model).config.pad_token_id,
            do_sample=True, top_p=0.9, temperature=0.5
        )
        prompt_encoding = generation_batch.prompt_encoding.to(args.device)
        output_sequences = tokenizer.batch_decode(args.accelerator.unwrap_model(model).generate(
            **prompt_encoding,
            generation_config=generation_config
        ), skip_special_tokens=True)

        if wandb_table is not None:
            for sequence in output_sequences:
                wandb_table.add_data(epoch, sequence)
            generation_table = wandb.Table(columns=wandb_table.columns, data=wandb_table.data)
            args.accelerator.log({"Generation": generation_table})

    return avg_ppl, generation_table


def main(args):
    set_seed(args.seed)

    # Initialize wandb
    # args.accelerator.init_trackers(
    #     "Impossible-Distillation",
    #     config=args,
    # )

    args.accelerator.init_trackers("Impossible-Distillation-Dialog", 
                              config={}, 
                              init_kwargs={"wandb":{"name":args.model_name}})

    with args.accelerator.main_process_first():
        run = args.accelerator.get_tracker("wandb") if not args.disabled else None
        wandb_table = wandb.Table(columns=["Epoch", "Generation"]) if not args.disabled else None

    if args.accelerator.is_main_process and not args.disabled:
        # Experiment directories
        args.ckpt_dir = os.path.join(args.save_dir, run.run.name)

        # Main routine
        if args.save_ckpt:
            os.makedirs(args.ckpt_dir, exist_ok=False)
            args.accelerator.print(f"Model Saving Directory: {args.ckpt_dir}")

    # Initialize tokenizer, model and optimizer
    tokenizer, model = init_tokenizer_and_model(args)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    # Initialize datasets
    train_dataset = CTGDataset.from_file(tokenizer, args.train_filename)
    valid_dataset = CTGDataset.from_file(tokenizer, args.valid_filename)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=CTGDataset.collate_fn, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                  collate_fn=CTGDataset.collate_fn, num_workers=4)

    # Prepare with accelerator
    model, optimizer, train_dataloader, valid_dataloader = args.accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )

    best_ppl = 1000
    ppl, wandb_table = evaluate(model, valid_dataloader, tokenizer, args, 0, wandb_table)
    for epoch in range(1, args.num_epochs + 1):
        train_one_epoch(model, train_dataloader, optimizer, args, epoch)

        ppl, wandb_table = evaluate(model, valid_dataloader, tokenizer, args, epoch, wandb_table)

        if args.accelerator.is_main_process and ppl < best_ppl:
            best_ppl = ppl

            if args.save_ckpt:
                print("Location of Saved Model: ", args.ckpt_dir)
                args.accelerator.unwrap_model(model).save_pretrained(args.ckpt_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # Run with CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --multi_gpu main.py --save_ckpt













