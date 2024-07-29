import argparse
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import pickle
import numpy as np
import wandb

accuracy_metric = evaluate.load("accuracy")

def preprocess_function(examples, tokenizer, max_input_length):
    return tokenizer(examples["sentence"], max_length=max_input_length, truncation=True, padding="max_length")

def evaluate_model(model, tokenizer, dataset, device, max_input_length):
    model.eval()
    predictions, references = [], []
    wandb_table = wandb.Table(columns=["Index", "Text", "Actual Label", "Predicted Label"])

    for index, sample in enumerate(tqdm(dataset)):
        # Preprocess inputs and labels
        preprocessed = preprocess_function({"sentence": sample["sentence"]}, tokenizer, max_input_length)
        input_ids = torch.tensor(preprocessed["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(preprocessed["attention_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        # Collect predictions and references
        predictions.append(prediction)
        references.append(sample["label"])

        # Add to wandb table for visualization
        if index < 10:
            print(f"Sample {index + 1}:")
            print(f"Text: {sample['sentence']}")
            print(f"Actual Label: {sample['label']}")
            print(f"Predicted Label: {prediction}\n")

        wandb_table.add_data(index, sample["sentence"], sample["label"], prediction)

    # Compute accuracy
    accuracy_score = accuracy_metric.compute(predictions=predictions, references=references)

    # Log results to WandB
    wandb.log({"evaluation_table": wandb_table})

    return accuracy_score


def main(args):
    wandb.init(project="Evaluation_SST2", entity="ivolinengong", name=args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_input_length = 128

    model = RobertaForSequenceClassification.from_pretrained(args.model_path).to(device)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    test_dataset = load_dataset("glue", "sst2", split="validation")

    print(f"\nMODEL: {args.model_path}\n")

    model.cuda()
    model.to(device)
    model.eval()

    metrics_results = evaluate_model(model, tokenizer, test_dataset, device, max_input_length)

    # Log aggregated metrics to wandb
    wandb_summary_metrics = {}
    if 'accuracy' in metrics_results:
        accuracy = metrics_results['accuracy']
        wandb_summary_metrics['accuracy'] = accuracy
        print(f'Accuracy: {accuracy:.2f}')

    wandb.summary.update(wandb_summary_metrics)
    wandb.finish()
    print("Evaluation completed and logged to wandb.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="path_to_your_model")
    parser.add_argument("--run_name", default="dp_evaluation")
    args = parser.parse_args()

    main(args)
