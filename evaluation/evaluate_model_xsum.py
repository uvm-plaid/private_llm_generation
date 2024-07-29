import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import pickle
import numpy as np

# Metric
# metric = evaluate.load("rouge")
# Load additional metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")

def word_count_filter(example):
    return len(example["document"].split()) <= 300

# def preprocess_function(examples):
#     inputs = ["summarize: " + doc for doc in examples["document"]]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#     labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")

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

def preprocess_function(examples, tokenizer, max_input_length, max_output_length):
    inputs = ["summarize: " + doc for doc in examples["document"]]
    # model_inputs = tokenizer(inputs, max_length=300, truncation=True, padding="max_length")
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=max_output_length, truncation=True, padding="max_length")
    
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


def evaluate_model(model, tokenizer, dataset, max_input_length, max_output_length):
    wandb_table = wandb.Table(columns=["Index", "Original Text", "Actual Summary", "Predicted Summary"])

    predictions, references = [], []
    for index, sample in enumerate(tqdm(dataset)):
        preprocessed = preprocess_function({"document": [sample["document"]], "summary": [sample["summary"]]}, tokenizer, max_input_length, max_output_length)
        input_ids = torch.tensor(preprocessed["input_ids"]).to(model.device)
        attention_mask = torch.tensor(preprocessed["attention_mask"]).to(model.device)

        with torch.no_grad():
            # outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_output_length)

            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=100, min_length=30, num_beams=5,
                                 length_penalty=2.0, no_repeat_ngram_size=2, early_stopping=True)
       
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(sample["summary"])

        if index < 10:
            print(f"Sample {index+1}:")
            print(f"Actual Summary: {sample['summary']}")
            print(f"Predicted Summary: {prediction}\n")
            print("---\n")

        # Add data to wandb table
        wandb_table.add_data(index, sample["document"], sample["summary"], prediction)
        # wandb_table.add_data(index, sample["dialogue"], sample["summary"], prediction)

    # Compute metrics
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    meteor_score = meteor.compute(predictions=predictions, references=references)
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=[[ref] for ref in references])

    wandb.log({"evaluation_table": wandb_table})
    
    return {
        "rouge": rouge_scores,
        "bertscore": bert_scores,
        "bleu": bleu_score,
        "meteor": meteor_score,
        "sacrebleu": sacrebleu_score
    }


import wandb

def main(args):
    # wandb.init(project="your_project_name", entity="your_wandb_entity")
    wandb.init(project="Evalutation_NEWS", entity="ivolinengong", name=args.run_name)

    # wandb.init(project="Evalutation_DIALOG", entity="ivolinengong", name="stage_dp_large18")
    # test_dataset = load_dataset("samsum", split="test")

    test_dataset = load_dataset("xsum", split="test")
    # filtered_test_dataset = test_dataset.filter(word_count_filter)

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_dataset = load_dataset("knkarthick/dialogsum", split="test")

    # model_path = "/users/k/n/kngongiv/Research/private_llm_generation/dialogue/dp/dp_results/dp"

    # model_path = "/users/k/n/kngongiv/Research/private_llm_generation/dialogue/dp/dp_results/dp_2"
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("t5-large")

    print(f"\nMODEL: {args.model_path}\n" )

    model.cuda()
    model.to(device)
    model.eval()

    max_input_length = 250
    # max_output_length = 150
    max_output_length = 80

    metrics_results = evaluate_model(model, tokenizer, test_dataset, max_input_length, max_output_length)
    #metrics_results = evaluate_model(model, tokenizer, filtered_test_dataset, max_input_length, max_output_length)  

    # Initialize dictionary for wandb summary metrics
    wandb_summary_metrics = {}

    # Handle ROUGE metrics
    if 'rouge' in metrics_results:
        rouge_scores = metrics_results['rouge']
        for key, value in rouge_scores.items():
            if isinstance(value, dict) and 'fmeasure' in value:  # Check for the expected structure
                wandb_summary_metrics[f'{key}_fmeasure'] = round(value['fmeasure'] * 100, 2)
                print(f'{key} F-measure: {value["fmeasure"] * 100:.2f}%')
            elif isinstance(value, np.float64):  # Directly dealing with numerical values
                wandb_summary_metrics[key] = round(value * 100, 2)
                print(f'{key}: {value * 100:.2f}%')

    # Handle BERTScore metrics
    if 'bertscore' in metrics_results:
        bert_scores = metrics_results['bertscore']
        for score_type in ['precision', 'recall', 'f1']:
            wandb_summary_metrics[f'bertscore_{score_type}'] = np.mean(bert_scores[score_type])

    # For metrics like BLEU, Meteor, SacreBLEU, ensure they return a 'score'
    for metric_name in ['bleu', 'meteor', 'sacrebleu']:
        if metric_name in metrics_results and 'score' in metrics_results[metric_name]:
            score = metrics_results[metric_name]['score']
            wandb_summary_metrics[metric_name] = score
            print(f'{metric_name}: {score:.2f}')

    # Log aggregated metrics to wandb
    wandb.summary.update(wandb_summary_metrics)
    
    # Log the table as well
    # wandb.log({"evaluation_table": wandb_table})

    wandb.finish()
    print("Evaluation completed and logged to wandb.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="path_to_your_model")
    parser.add_argument("--run_name", default="dp")
    args = parser.parse_args()

    main(args)
