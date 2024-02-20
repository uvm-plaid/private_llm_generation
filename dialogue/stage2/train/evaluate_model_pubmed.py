import numpy as np
import evaluate

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainerCallback

import pandas as pd
from IPython.display import display, HTML
from datasets import load_metric
import numpy as np

import torch
import accelerate
import evaluate
import os
import wandb

import spacy
spacy_model = spacy.load("en_core_web_sm")

# # Download necessary NLTK models
nltk.download("punkt", quiet=True)

def preprocess_function(examples):
    def get_first_5_sentences(text):
        doc = spacy_model(text)
        sentences = [sent.text.strip() for sent in doc.sents][:5]
        return " ".join(sentences)
    
    original_texts = [get_first_5_sentences(doc) for doc in examples["article"]]
    inputs = ["summarize: " + doc for doc in original_texts] 
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["abstract"], max_length=max_output_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["original_texts"] = original_texts  # Store original texts for later use
    
    # # Calculate lengths
    # input_lengths = [len(input_id) for input_id in model_inputs["input_ids"]]
    return model_inputs


def compute_metrics(predictions, references):
    metrics_result = {}
    
    # Compute ROUGE
    rouge_result = rouge.compute(predictions=predictions, references=references)
    for key in ['rouge1', 'rouge2', 'rougeL']:
        metrics_result[f"rouge_{key}"] = rouge_result[key]

    # Compute BERTScore
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang='en')
    metrics_result.update({
        'bertscore_precision': np.mean(bertscore_result['precision']),
        'bertscore_recall': np.mean(bertscore_result['recall']),
        'bertscore_f1': np.mean(bertscore_result['f1'])
    })

    # BLEU, METEOR, SacreBLEU (as example, add others similarly)
    metrics_result['bleu'] = bleu.compute(predictions=predictions, references=references)['bleu']
    metrics_result['meteor'] = meteor.compute(predictions=predictions, references=references)['meteor']
    metrics_result['sacrebleu'] = sacrebleu.compute(predictions=predictions, references=references)['score']

    return metrics_result


# Load additional metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")

# wandb.init(project="StageDP-Evalutation_PubMed", entity="ivolinengong")
wandb.init(project="Final-Evalutation_PubMed", entity="ivolinengong")
# wandb_table = wandb.Table(columns=["Index", "Original Text", "Actual Summary", "Predicted Summary", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore Precision", "BERTScore Recall", "BERTScore F1", "BLEU", "METEOR", "SacreBLEU"])
wandb_table = wandb.Table(columns=["Index", "Original Text", "Actual Summary", "Predicted Summary"])

# max_input_length = 1024
max_input_length = 250 #150
max_output_length = 100 #100 
# tokenizer = AutoTokenizer.from_pretrained("t5-large")
tokenizer = AutoTokenizer.from_pretrained("t5-large") #, model_max_length=max_input_length)

dataset = load_dataset("scientific_papers", "pubmed", split="test")
# dataset = load_dataset("scientific_papers", "pubmed", split="test[:200]")
tokenized_test_dataset = dataset.map(preprocess_function, batched=True)

# Load the model
# model_path = "/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage_dp/model/sage-glade-19"  # StageDP Model
model_path = "/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage2/train/gen_model/vibrant-dragon-11"  # Final Model

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
if torch.cuda.is_available():
    model.cuda()

# Evaluation
model.eval()

num_samples_to_print = 10
all_metrics_scores = []


for i, batch in enumerate(tokenized_test_dataset):
    # if i >= 200:  # Break the loop after evaluating 200 samples
    #     break
    with torch.no_grad():
        inputs = tokenizer(batch['article'], return_tensors="pt", padding=True, truncation=True, max_length=3200)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    

        outputs = model.generate(
            inputs['input_ids'], 
            max_length=100,  # Maximum summary length in tokens
            min_length=30,  # Minimum summary length to ensure summaries aren't too short
            num_beams=5,  # Beam search helps in generating more coherent summaries
            length_penalty=2.0,  # Encourages the model to utilize the maximum length
            no_repeat_ngram_size=2,  # Prevents repeating n-grams
            early_stopping=True  # Stops generation when all beams generate EOS token
        )

        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_label = tokenizer.decode(batch['labels'], skip_special_tokens=True)
        original_text = batch["original_texts"]

        if i < num_samples_to_print:
            print(f"Original Text: {original_text}\n")
            print(f"Actual Summary:{decoded_label}\n", flush=True)
            print(f"Predicted Summary: {decoded_pred}\n", flush=True)
            print("---", flush=True)

        # Compute all metrics for this batch
        metrics_result = compute_metrics(predictions=[decoded_pred], references=[decoded_label])
        all_metrics_scores.append(metrics_result)

        # Add data to wandb table
        wandb_table.add_data(
            i,
            original_text,
            decoded_label,
            decoded_pred
            # metrics_result["rouge_rouge1"],
            # metrics_result["rouge_rouge2"],
            # metrics_result["rouge_rougeL"],
            # metrics_result["bertscore_precision"],
            # metrics_result["bertscore_recall"],
            # metrics_result["bertscore_f1"],
            # metrics_result["bleu"],
            # metrics_result["meteor"],
            # metrics_result["sacrebleu"]
        )

# Compute average scores for all metrics
average_metrics_scores = {metric: np.mean([score[metric] for score in all_metrics_scores]) for metric in all_metrics_scores[0]}
print(f"Average Metrics Scores:{average_metrics_scores}" , flush=True)

wandb.log({"evaluation_results": wandb_table}) #, "average_metrics": average_metrics_scores})
# Log all average metrics at once to the summary
wandb.run.summary.update({"average_metrics": average_metrics_scores})

# We are only getting the 1st 50 tokens from the abstract as the label, is this  a problem?

