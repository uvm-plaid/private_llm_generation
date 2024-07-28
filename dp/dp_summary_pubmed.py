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
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # Prepare labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["abstract"], max_length=max_output_length, padding="max_length", truncation=True)

    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    return model_inputs

# def preprocess_function(examples):
#     # Modified part for sentence selection
#     def get_first_3_sentences(text):
#         doc = spacy_model(text)
#         sentences = [sent.text.strip() for sent in doc.sents][:3]
#         return " ".join(sentences)
    
#     inputs = ["summarize: " + get_first_3_sentences(doc) for doc in examples["article"]]
#     model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

#     # Prepare labels
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(examples["abstract"], max_length=max_output_length, padding="max_length", truncation=True)

#     model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
#     model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
#     model_inputs["labels"] = torch.tensor(labels["input_ids"])

#     # Calculate lengths
#     input_lengths = [len(input_id) for input_id in model_inputs["input_ids"]]
    
#     # Add lengths to model_inputs dictionary
#     model_inputs["lengths"] = input_lengths
    
#     return model_inputs

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

# wandb.init(project="Summarize_DP_PubMed", entity="ivolinengong")
wandb.init(project="DP_Summarization_PubMed", entity="ivolinengong", name="pubmed_dp_large")



# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(42)

# Load the tokenizer
max_input_length = 512 #1024
max_output_length = 150 #256 #250
batch_size = 3
epochs = 3
gradient_accumulation_steps = 10
# target_epsilon = 8
target_epsilon = 10000000 #0.000001 #10000000 # #0.000001 #8

# Load pubmed dataset
# train_dataset = load_dataset("scientific_papers", "pubmed", split="train[:5%]", cache_dir="./pubmed_data/")
# val_dataset = load_dataset("scientific_papers", "pubmed", split="validation[:16%]", cache_dir="./pubmed_data/")

train_dataset = load_dataset("scientific_papers", "pubmed", split="train", cache_dir="./pubmed_data/")
val_dataset = load_dataset("scientific_papers", "pubmed", split="validation", cache_dir="./pubmed_data/")
# test_dataset = load_dataset("scientific_papers", "pubmed", split="test", cache_dir="./pubmed_data/")

# tokenized_datasets = dataset.map(preprocess_function, batched=True)
# train_dataset = tokenized_datasets["train")

# Checkpoint directories
# checkpoint_dir = "./results/dp_results/test/run"
checkpoint_dir = f"./results/dp_results/{wandb.run.name}"
# checkpoint_dir = f"./results/dp_results/test/rich-serenity-9"
os.makedirs(checkpoint_dir, exist_ok=True)

model_tokenizer_path = checkpoint_dir
optimizer_checkpoint_path = os.path.join(checkpoint_dir, "optimizer_and_loss.pth")

# Load model and tokenizer, and optimizer state if available
# model_checkpoint = find_latest_checkpoint(checkpoint_dir)
# if model_checkpoint:
#     print(f"Resuming from checkpoint: {model_checkpoint}")
#     # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
#     # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#     model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
#     tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=max_input_length)

#     if os.path.isfile(optimizer_checkpoint_path):
#         print("Loading optimizer state and best loss from checkpoint")
#         checkpoint = torch.load(optimizer_checkpoint_path, map_location=device)
#         optimizer = AdamW(model.parameters(), lr=5e-5)
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         best_loss = checkpoint['best_loss']
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

# Tokenization of dataset
# tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=max_length)
# tokenizer = AutoTokenizer.from_pretrained("t5-large", model_max_length=max_input_length)
    

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=5)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, num_proc=5)
# tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Vanilla_Summary2 ; max_input_length => 1024")
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the T5 model
# model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-large").to(device)

# Initialize the optimizer
# optimizer = AdamW(model.parameters(), lr=5e-5)

# Attach the privacy engine to the optimizer
privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(tokenized_train_dataset),
    epochs=epochs,
    max_grad_norm=1.0,
    target_epsilon=target_epsilon, #3,
    noise_multiplier=0.1,
    # clipping_mode="ghost",
)
privacy_engine.attach(optimizer)

print(f"Privacy Parameters -> Epsilon: {privacy_engine.target_epsilon}, "
      f"Noise Multiplier: {privacy_engine.noise_multiplier}, "
      f"Clipping Norm: {privacy_engine.max_grad_norm}, "
      f"Delta: {privacy_engine.target_delta}")

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

        # # Optimization step (without calling loss.backward())
        # optimizer.step(loss=loss)

        # optimizer.zero_grad()
        # scheduler.step()
        
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
        # Log metrics to wandb
        # wandb.log({"Epoch": epoch + 1, "Step": i, "Loss": avg_loss})
        
        # print(f"Epoch: {epoch + 1}, Step: {i}/{len(data_loader)}, Loss: {avg_loss}\n")

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
        # if i % 10 == 0:
        #     print(f"Epoch: {epoch + 1}, Step: {i}/{len(data_loader)}, Loss: {avg_loss}", flush=True)

        if i % 10 == 0:
            privacy_spent = privacy_engine.get_privacy_spent()
            # epsilon = privacy_spent['epsilon']
            print(f"Epoch: {epoch + 1}, Step: {i}/{len(data_loader)}, Loss: {avg_loss}, Epsilon: {privacy_spent}", flush=True)

            # Calculate the average training loss and perplexity for the epoch
    avg_train_loss = total_loss / len(data_loader)
    train_perplexity = np.exp(avg_train_loss)
    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_perplexity": train_perplexity})
    
    # # Validation loop
    # model.eval()
    # total_eval_loss = 0
    # with torch.no_grad():
    #     for batch in val_data_loader:
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)

    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         logits = outputs.logits

    #         # Compute per-example loss
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none").mean(dim=1)

    #         # Accumulate the validation loss
    #         total_eval_loss += torch.mean(loss).item()

    # # Calculate the average validation loss and perplexity
    # avg_val_loss = total_eval_loss / len(tokenized_val_dataset)
    # val_perplexity = np.exp(avg_val_loss)
    # wandb.log({"epoch": epoch + 1,"val_loss": avg_val_loss, "val_perplexity": val_perplexity})

# # Save the final model
# final_model_path = os.path.join(checkpoint_dir, "t5_pubmed_dp_model_final.pth")
# torch.save(model.state_dict(), final_model_path)
# print(f"Final model saved at {final_model_path}")

wandb.finish()


# NOTES
# DP Summary works with 1024 for the whole dataset
