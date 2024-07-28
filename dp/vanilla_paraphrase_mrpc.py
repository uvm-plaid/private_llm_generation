from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from transformers import TrainerCallback

import datasets
import random
import pandas as pd
from IPython.display import display, HTML
from datasets import load_metric
import numpy as np
import wandb

import torch
import accelerate
import evaluate
import os

class PerplexityCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        perplexity = np.exp(state.log_history[-1]["loss"])
        wandb.log({"train_perplexity": perplexity}, step=state.global_step)

class ExtendedPerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Check if 'loss' or 'eval_loss' is in logs
        if 'loss' in logs:
            # Calculate training perplexity and log it
            train_perplexity = np.exp(logs['loss'])
            wandb.log({"train_perplexity": train_perplexity}, step=state.global_step)
        elif 'eval_loss' in logs:
            # Calculate validation perplexity and log it
            val_perplexity = np.exp(logs['eval_loss'])
            wandb.log({"val_perplexity": val_perplexity, "val_loss": logs['eval_loss']}, step=state.global_step)

# def preprocess_function(examples):
#     inputs = ["paraphrase: " + sentence for sentence in examples["sentence1"]]
#     model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
#     labels = tokenizer(examples["sentence2"], max_length=128, truncation=True, padding="max_length")

#     # Ensure labels are prepared for loss calculation
#     labels["input_ids"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in label]
#         for label in labels["input_ids"]
#     ]

#     return {
#         "input_ids": model_inputs["input_ids"],
#         "attention_mask": model_inputs["attention_mask"],
#         "labels": labels["input_ids"]
#     }

def preprocess_function(examples):
    inputs = ["paraphrase: " + sentence for sentence in examples["sentence1"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(examples["sentence2"], max_length=128, truncation=True, padding="max_length")

    # Prepare labels for loss calculation: replace padding token id's in the labels with -100
    labels["input_ids"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels["input_ids"]
    ]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"]
    }


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not checkpoint_subdirs:
        return None
    latest_checkpoint = max(checkpoint_subdirs, key=lambda d: int(d.split('-')[-1]))
    return os.path.join(checkpoint_dir, latest_checkpoint)


# Set seeds for reproducibility
torch.manual_seed(42)

# Initialize Weights & Biases
wandb.init(project="Paraphrasing_MRPC", entity="ivolinengong", name="vanilla_full")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)

checkpoint_dir = f"./{wandb.run.name}"
os.makedirs(checkpoint_dir, exist_ok=True)
model_checkpoint = find_latest_checkpoint(checkpoint_dir)

if model_checkpoint:
    print(f"Resuming from checkpoint: {model_checkpoint}")
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
else:
    print("Starting training from the pretrained model")
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large")

if torch.cuda.is_available():
    model.cuda()

# Load the tokenizer
max_input_length = 128 #300 #1024
max_output_length = 128 #256 #250
batch_size = 4
epochs = 3
gradient_accumulation_steps = 32 #10
target_epsilon = 8

# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")
train_dataset = dataset['train'].map(preprocess_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

eval_dataset = dataset['validation'].map(preprocess_function, batched=True)
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("\nTraining dataset size:", len(train_dataset))
# print("Evaluation dataset size:", len(eval_dataset))

# # Prepare model
# model = T5ForConditionalGeneration.from_pretrained("t5-base")
# model.to(device)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_dir, #"./mrpc_checkpoint",
    evaluation_strategy="epoch",
    learning_rate=5e-5,  
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2,
    weight_decay=0.01, 
    save_total_limit=3,
    num_train_epochs=epochs,  
    report_to="wandb",
    gradient_accumulation_steps=8,  
    resume_from_checkpoint=model_checkpoint if model_checkpoint else None
    # per_device_train_batch_size=16, 
    # weight_decay=0.01, 
    # num_train_epochs=3,  
    # report_to="wandb"
)

# Define a trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[ExtendedPerplexityCallback()]
)

# Start training
trainer.train()
