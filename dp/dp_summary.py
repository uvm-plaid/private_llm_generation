import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import os
import wandb
import spacy

# Load the SpaCy model for sentence tokenization
spacy_model = spacy.load("en_core_web_sm")

def preprocess_function(examples, tokenizer, max_input_length, max_output_length, dataset_name):
    if dataset_name == "xsum":
        inputs = ["summarize: " + doc for doc in examples["document"]]
        labels = tokenizer(examples["summary"], max_length=max_output_length, truncation=True, padding="max_length")
    elif dataset_name == "pubmed":
        inputs = ["summarize: " + doc for doc in examples["article"]]
        labels = tokenizer(examples["abstract"], max_length=max_output_length, truncation=True, padding="max_length")
    else:
        raise ValueError("Unsupported dataset specified")

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

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

def main(args):
    wandb.init(project=args.project_name, entity=args.entity, name=args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Load dataset
    if args.dataset_name == "xsum":
        dataset = load_dataset("xsum", split="train")
    elif args.dataset_name == "pubmed":
        dataset = load_dataset("scientific_papers", "pubmed", split="train", cache_dir="./pubmed_data/")
    else:
        raise ValueError("Unsupported dataset_name specified")

    # # Use only a subset of the training dataset for testing
    # if args.subset_size:
    #     train_dataset = dataset.select(range(args.subset_size))
    # else:
    #     train_dataset = dataset

    tokenized_train_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_input_length, args.max_output_length, args.dataset_name),
        batched=True
    )
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print("\nTraining dataset size:", len(dataset))
    print("\nFiltered Training dataset size:", len(tokenized_train_dataset))

    sample = tokenized_train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")

    checkpoint_dir = f"./models/{wandb.run.name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if model_checkpoint:
        print(f"Resuming from checkpoint: {model_checkpoint}")

        # Load model and tokenizer from the checkpoint
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

        # Load optimizer state
        if os.path.isfile(args.optimizer_checkpoint_path):
            print("Loading optimizer state and best loss from checkpoint")
            checkpoint = torch.load(args.optimizer_checkpoint_path, map_location=device)
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
    else:
        print("Starting training from the pretrained model")
        model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        best_loss = float('inf')

    # Attach the privacy engine to the optimizer
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=len(tokenized_train_dataset),
        epochs=args.epochs,
        max_grad_norm=1.0,
        target_epsilon=args.target_epsilon,
    )
    privacy_engine.attach(optimizer)

    print(f"Privacy Parameters -> Epsilon: {privacy_engine.target_epsilon}, "
          f"Noise Multiplier: {privacy_engine.noise_multiplier}, "
          f"Clipping Norm: {privacy_engine.max_grad_norm}, "
          f"Delta: {privacy_engine.target_delta}")

    # Learning rate scheduler
    total_steps = len(tokenized_train_dataset) // args.batch_size * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Data loader
    data_loader = DataLoader(tokenized_train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}", flush=True)
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
            per_example_loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none").mean(dim=1)

            # Gradient accumulation
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step(loss=per_example_loss)  # Perform parameter update
                optimizer.zero_grad()  # Reset gradients
                scheduler.step()
            else:
                optimizer.virtual_step(loss=per_example_loss)  # Accumulate gradients

            avg_loss = torch.mean(per_example_loss).item()
            total_loss += avg_loss * len(per_example_loss)

            # Save the model if this is the best loss so far
            if avg_loss < best_loss:
                best_loss = avg_loss

                # Save model and tokenizer using save_pretrained
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                # Save optimizer state and best_loss separately
                torch.save({
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, args.optimizer_checkpoint_path)

                print(f"New best model saved with loss: {best_loss}")

            if i % 10 == 0:
                privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=True)
                print(f"Epoch: {epoch + 1}, Step: {i}/{len(data_loader)}, Loss: {avg_loss}, "
                      f"Epsilon: {privacy_spent}", flush=True)

        # Calculate the average training loss and perplexity for the epoch
        avg_train_loss = total_loss / len(data_loader)
        train_perplexity = np.exp(avg_train_loss)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_perplexity": train_perplexity})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help="Name of the run for WandB logging.")
    parser.add_argument("--project_name", type=str, default="DP_Summarization", help="Name of the WandB project.")
    parser.add_argument("--entity", type=str, default="ivolinengong", help="WandB entity name.")
    parser.add_argument("--model_name", type=str, default="t5-large", help="Name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["xsum", "pubmed"], help="Dataset name.")
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum input length for the tokenizer.")
    parser.add_argument("--max_output_length", type=int, default=150, help="Maximum output length for the tokenizer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--target_epsilon", type=float, default=8, help="Target epsilon for differential privacy.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--subset_size", type=int, default=None, help="Size of the subset of the training data to use.")
    parser.add_argument("--optimizer_checkpoint_path", type=str, default="./optimizer_and_loss.pth", help="Path to save the optimizer checkpoint.")

    args = parser.parse_args()

    main(args)
