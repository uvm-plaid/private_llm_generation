import argparse
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import os
import wandb
import spacy

# Load the SpaCy model for sentence tokenization
spacy_model = spacy.load("en_core_web_sm")

# def preprocess_function(examples):
def preprocess_function(examples, tokenizer, max_input_length):
    model_inputs = tokenizer(examples["sentence"], max_length=args.max_input_length, truncation=True, padding="max_length")
    labels = examples["label"]
    model_inputs["labels"] = labels
    return model_inputs

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not checkpoint_subdirs:
        return None
    latest_checkpoint = max(checkpoint_subdirs, key=lambda d: int(d.split('-')[-1]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def main(args):
    wandb.init(project="SST2_DP", entity="ivolinengong", name=args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name).to(device)

    # Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # Use only a subset of the training dataset for testing
    if args.subset_size:
        train_dataset = dataset['train'].select(range(args.subset_size))
    else:
        train_dataset = dataset['train']

    # tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer, args.max_input_length), batched=True)
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print("\nTraining dataset size:", len(train_dataset))
    print("\nFiltered Training dataset size:", len(tokenized_train_dataset))

    sample = tokenized_train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels: {sample['labels']}")

    checkpoint_dir = f"./models/{wandb.run.name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if model_checkpoint:
        print(f"Resuming from checkpoint: {model_checkpoint}")

        # Load model and tokenizer from the checkpoint
        model = RobertaForSequenceClassification.from_pretrained(model_checkpoint).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

        # Load optimizer state
        if os.path.isfile(args.optimizer_checkpoint_path):
            print("Loading optimizer state and best loss from checkpoint")
            checkpoint = torch.load(args.optimizer_checkpoint_path, map_location=device)
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
    else:
        print("Starting training from the pretrained model")
        model = RobertaForSequenceClassification.from_pretrained(args.model_name).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
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
        correct_predictions = 0
        total_predictions = 0
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}", flush=True)
        for i, batch in enumerate(data_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, reduction='none')

            # Compute accuracy
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            # Gradient accumulation
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step(loss=loss)  # Perform parameter update
                optimizer.zero_grad()  # Reset gradients
                scheduler.step()
            else:
                optimizer.virtual_step(loss=loss)  # Accumulate gradients
                print("Accumulating Grads: ", args.gradient_accumulation_steps)

            avg_loss = loss.mean().item()
            total_loss += avg_loss * len(loss)

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
                training_stats = optimizer.get_training_stats()
                print(f"Epoch: {epoch + 1}, Step: {i}/{len(data_loader)}, Loss: {avg_loss}, "
                      f"Epsilon: {privacy_spent}, Training stats: {training_stats}", flush=True)

        # Calculate the average training loss and accuracy for the epoch
        avg_train_loss = total_loss / len(data_loader)
        train_accuracy = correct_predictions.double() / total_predictions
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_accuracy": train_accuracy,
                   "privacy_spent": privacy_spent, "training_stats": training_stats})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help="Name of the run for WandB logging.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the model to use.")
    parser.add_argument("--max_input_length", type=int, default=128, help="Maximum input length for the tokenizer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Number of steps for gradient accumulation.")
    parser.add_argument("--target_epsilon", type=float, default=20, help="Target epsilon for differential privacy.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--subset_size", type=int, default=None, help="Size of the subset of the training data to use.")
    parser.add_argument("--optimizer_checkpoint_path", type=str, default="./optimizer_and_loss.pth", help="Path to save the optimizer checkpoint.")

    args = parser.parse_args()

    main(args)
