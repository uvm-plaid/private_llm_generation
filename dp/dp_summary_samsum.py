import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from datasets import load_dataset
from torch.utils.data import DataLoader
from private_transformers import PrivacyEngine
import torch.nn.functional as F
# from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW  
import numpy as np
import os
import wandb
import spacy

# Load the SpaCy model for sentence tokenization
spacy_model = spacy.load("en_core_web_sm")


def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # 'summary' field is used as labels in SAMSum
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_output_length, padding="max_length", truncation=True)

    # Replace padding token IDs in labels with -100 to ignore them in loss calculation
    labels["input_ids"] = [
        [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in label]
        for label in labels["input_ids"]
    ]

    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    return model_inputs


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not checkpoint_subdirs:
        return None
    latest_checkpoint = max(checkpoint_subdirs, key=lambda d: int(d.split('-')[-1]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def compute_average_length(dataset):
    lengths = dataset["lengths"]  # Assuming 'lengths' are stored in the dataset
    average_length = np.mean(lengths)
    print(f"Average token length: {average_length}")

wandb.init(project="SamSum_Summarization", entity="ivolinengong", name="samsum_dp")


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(42)

# Load the tokenizer
max_input_length = 150 #1024
max_output_length = 100 #256 #250
batch_size = 3
epochs = 8
gradient_accumulation_steps = 10

# Load SAMSum dataset
train_dataset = load_dataset("samsum", split="train")
val_dataset = load_dataset("samsum", split="validation")

# Checkpoint directories
# checkpoint_dir = f"./dp_results/{wandb.run.name}"
checkpoint_dir = f"./samsum_results/{wandb.run.name}"
os.makedirs(checkpoint_dir, exist_ok=True)

model_tokenizer_path = checkpoint_dir
optimizer_checkpoint_path = os.path.join(checkpoint_dir, "optimizer_and_loss.pth")

# Load model and tokenizer, and optimizer state if available
model_checkpoint = find_latest_checkpoint(checkpoint_dir)
if model_checkpoint:
    print(f"Resuming from checkpoint: {model_checkpoint}")

    # Load model and tokenizer from the checkpoint
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, model_max_length=max_input_length)

    # Load optimizer state
    if os.path.isfile(optimizer_checkpoint_path):
        print("Loading optimizer state and best loss from checkpoint")
        checkpoint = torch.load(optimizer_checkpoint_path, map_location=device)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
else:
    print("Starting training from the pretrained model")
    model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=max_input_length)

    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-large").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("t5-large", model_max_length=max_input_length)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_loss = float('inf')

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


print("\nTraining dataset size:", len(tokenized_train_dataset))
print("Validation dataset size:", len(tokenized_val_dataset))


# # Inspect a sample
sample = tokenized_train_dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Attention mask shape: {sample['attention_mask'].shape}")
print(f"Labels shape: {sample['labels'].shape}")
# Calculate and print average token length
# compute_average_length(tokenized_train_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attach the privacy engine to the optimizer
privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(tokenized_train_dataset),
    epochs=epochs,
    max_grad_norm=1.0,
    target_epsilon=8,
    noise_multiplier=0.1,
    # clipping_mode="ghost",
)
privacy_engine.attach(optimizer)

# Learning rate scheduler
total_steps = len(tokenized_train_dataset) // batch_size * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Data loader
data_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(tokenized_val_dataset, batch_size=batch_size)


# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    print(f"\nStarting epoch {epoch + 1}/{epochs}", flush=True)
    for i, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        # Compute per-example loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none").mean(dim=1)

        # Gradient accumulation
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step(loss=loss)  # Perform parameter update
            optimizer.zero_grad()  # Reset gradients
            scheduler.step()
        else:
            optimizer.virtual_step(loss=loss)  # Accumulate gradients


        # Save the model if this is the best loss so far
        avg_loss = torch.mean(loss).item()
        total_loss += avg_loss

        # Check if the current loss is the best
        if avg_loss < best_loss:
            best_loss = avg_loss

            # Save model and tokenizer using save_pretrained
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save optimizer state and best_loss separately
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, optimizer_checkpoint_path)

            print(f"New best model saved with loss: {best_loss}")

        # # Log progress
        if i % 10 == 0:
            print(f"Epoch: {epoch + 1}, Step: {i}/{len(data_loader)}, Loss: {avg_loss}", flush=True)

    # Calculate the average training loss and perplexity for the epoch
    avg_train_loss = total_loss / len(data_loader)
    train_perplexity = np.exp(avg_train_loss)
    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_perplexity": train_perplexity})

wandb.finish()
