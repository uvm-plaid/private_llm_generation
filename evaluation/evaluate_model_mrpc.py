import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import wandb
import numpy as np

# Load evaluation metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")

# Function to preprocess the dataset
def preprocess_function(examples, tokenizer, max_input_length, max_output_length):
    inputs = ["paraphrase: " + sentence for sentence in examples["sentence1"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(examples["sentence2"], max_length=max_output_length, truncation=True, padding="max_length")
    labels["input_ids"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels["input_ids"]]
    return {"input_ids": model_inputs["input_ids"], "attention_mask": model_inputs["attention_mask"], "labels": labels["input_ids"]}

# Function to evaluate the model
def evaluate_model(model, tokenizer, dataset, device, max_input_length, max_output_length):
    model.eval()
    predictions, references = [], []
    wandb_table = wandb.Table(columns=["Index", "Original Text", "Actual Paraphrase", "Predicted Paraphrase"])

    for index, sample in enumerate(tqdm(dataset)):
        # Preprocess inputs and labels
        preprocessed = preprocess_function({"sentence1": [sample["sentence1"]], "sentence2": [sample["sentence2"]]}, tokenizer, max_input_length, max_output_length)
        input_ids = torch.tensor(preprocessed["input_ids"]).to(device)
        attention_mask = torch.tensor(preprocessed["attention_mask"]).to(device)

        # Print the shapes of the tensors
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_length=max_output_length, 
                min_length=30, 
                num_beams=5, 
                length_penalty=2.0, 
                no_repeat_ngram_size=2, 
                early_stopping=True
            )

        # Decode and collect predictions and references
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(sample["sentence2"])

        # Add to wandb table for visualization
        if index < 10:
            print(f"Sample {index+1}:")
            print(f"Original: {sample['sentence1']}")
            print(f"Actual Paraphrase: {sample['sentence2']}")
            print(f"Predicted Paraphrase: {prediction}\n")
        
        wandb_table.add_data(index, sample["sentence1"], sample["sentence2"], prediction)

    # Compute metrics
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    meteor_score = meteor.compute(predictions=predictions, references=references)
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=[[ref] for ref in references])

    # Log results to WandB
    wandb.log({"evaluation_table": wandb_table})

    return {
        "rouge": rouge_scores,
        "bertscore": bert_scores,
        "bleu": bleu_score,
        "meteor": meteor_score,
        "sacrebleu": sacrebleu_score
    }

def main(args):
    wandb.init(project="Evaluation_MRPC", entity="ivolinengong", name=args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_output_length = 128

    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-large")

    test_dataset = load_dataset("glue", "mrpc", split="test")

    print(f"\nMODEL: {args.model_path}\n")

    max_input_length = 128
    
    metrics_results = evaluate_model(model, tokenizer, test_dataset, device, max_input_length, max_output_length)

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

    wandb.finish()
    print("Evaluation completed and logged to wandb.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="path_to_your_model")
    parser.add_argument("--run_name", default="dp")
    args = parser.parse_args()

    main(args)
