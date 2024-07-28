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

print(accelerate.__version__)


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


def word_count_filter(example):
    return len(example["document"].split()) <= 300

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")

    # Ensure labels are prepared for loss calculation
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

wandb.init(project="Vanilla_Summarization_News", entity="ivolinengong", name="vanilla_full")

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(42)

# Load the tokenizer
max_input_length = 512 #1024
max_output_length = 100 #256 #250
batch_size = 4
epochs = 3

# Tokenization of dataset
# tokenizer = AutoTokenizer.from_pretrained("t5-large", model_max_length=max_input_length)
tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=max_input_length)

train_dataset = load_dataset("xsum", split="train")
# filtered_train_dataset = train_dataset.filter(word_count_filter)
# # Apply preprocessing to the filtered dataset
# tokenized_train_dataset = filtered_train_dataset.map(preprocess_function, batched=True)
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("\nTraining dataset size:", len(tokenized_train_dataset))
# print("Validation dataset size:", len(tokenized_val_dataset))

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


training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_dir,
    evaluation_strategy="epoch",
    learning_rate=3e-5,  
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2,
    weight_decay=0.01, 
    save_total_limit=3,
    num_train_epochs=epochs,  
    report_to="wandb",
    gradient_accumulation_steps=8,  
    resume_from_checkpoint=model_checkpoint if model_checkpoint else None
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_val_dataset,
    callbacks=[ExtendedPerplexityCallback()]
)

trainer.train()


# # Load ROUGE metric
# # rouge = load_metric("rouge")
# rouge = evaluate.load("rouge")

# # Apply the Evaluation Function on the Test Dataset
# results = tokenized_test_dataset.map(evaluate, batched=True, batch_size=batch_size)

# # Calculate the mean of the results
# rouge1_avg = np.mean([score for result in results['rouge1'] for score in result])
# rouge2_avg = np.mean([score for result in results['rouge2'] for score in result])
# rougeL_avg = np.mean([score for result in results['rougeL'] for score in result])

# print(f"Average Rouge-1 score: {rouge1_avg}")
# print(f"Average Rouge-2 score: {rouge2_avg}")
# print(f"Average Rouge-L score: {rougeL_avg}")

# # Log the results to Weights & Biases
# wandb.log({"avg_rouge1": rouge1_avg, "avg_rouge2": rouge2_avg, "avg_rougeL": rougeL_avg})
# wandb.finish()



