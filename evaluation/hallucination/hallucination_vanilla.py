
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gliner import GLiNER
from transformers import pipeline

import argparse

# Load the GLiNER PII model
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

# Initialize the entailment model pipeline
entailment_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")

# Function to detect PII
def detect_pii(text, labels):
    try:
        entities = model.predict_entities(text, labels)
        print(f"Unextracted entities: {entities}")  # Print the unextracted entities
        extracted_entities = [{'text': entity['text'], 'label': entity['label']} for entity in entities]
        print(f"Extracted entities: {extracted_entities}")  # Print the extracted entities
        return extracted_entities
    except KeyError as e:
        print(f"KeyError: {e}")
        return []

# Define the PII labels
pii_labels = ["person", "organization", "phone number", "address", "passport number", "email", 
              "credit card number", "social security number", "health insurance id number", 
              "date of birth", "mobile phone number", "bank account number", "medication", 
              "cpf", "driver's license number", "tax identification number", "medical condition", 
              "identity card number", "national id number", "ip address", "email address", "iban", 
              "credit card expiration date", "username", "health insurance number", "registration number", 
              "student id number", "insurance number", "flight number", "landline phone number", "blood type", 
              "cvv", "reservation number", "digital signature", "social media handle", "license plate number", 
              "cnpj", "postal code", "passport number", "serial number", "vehicle registration number", 
              "credit card brand", "fax number", "visa number", "insurance company", "identity document number", 
              "transaction number", "national health insurance number", "cvc", "birth certificate number", 
              "train ticket number", "passport expiration date", "social_security_number"]

# Function to load data from JSON file
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Function to save PII detection results
def save_pii_results(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

# Function to load PII detection results
def load_pii_results(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return None



def check_entailment(original_text, candidate_text):
    """Check if candidate_text is entailed by original_text using the entailment model."""
    results = entailment_pipeline(f"{original_text} [SEP] {candidate_text}")
    for result in results:
        if result['label'] == 'ENTAILMENT' and result['score'] > 0.5:
            return True
    return False

# Updated function to compute changes using entailment
def compute_pii_changes_entailment(original, modified):
    added = []
    dropped = []

    original_texts = {item['text']: item for item in original}
    modified_texts = {item['text']: item for item in modified}

    # Check for dropped items using entailment
    for orig_text in original_texts:
        if not any(check_entailment(orig_text, mod_text) for mod_text in modified_texts):
            dropped.append(orig_text)

    # Check for added items using entailment
    for mod_text in modified_texts:
        if not any(check_entailment(mod_text, orig_text) for orig_text in original_texts):
            added.append(mod_text)

    return added, dropped


# Function to count PII entities that are added or dropped
def compute_pii_changes(original, modified):
    original_set = set([item['text'] for item in original])
    modified_set = set([item['text'] for item in modified])
    added = modified_set - original_set
    dropped = original_set - modified_set
    print("original: ", original)
    print("modified: ", modified)
    print(f"Original: {original_set}, Modified: {modified_set}, Added: {added}, Dropped: {dropped}")
    return list(added), list(dropped)

# Main function
def main(vanilla_file, ours_file, output_file, save_file):
    print("starts")
    # Load data
    vanilla_data = load_data(vanilla_file)
    ours_data = load_data(ours_file)

    # Extract instructions and outputs
    instructions = [item['instruction'] for item in vanilla_data]
    vanilla_outputs = [item['output'] for item in vanilla_data]
    ours_outputs = [item['output'] for item in ours_data]

    print("INFO EXTRACTED")
    # Load or detect PII
    pii_results = load_pii_results(save_file)
    if not pii_results:
        print("\nGenerating PII File")
        pii_instructions = [detect_pii(text, pii_labels) for text in instructions]
        pii_vanilla_outputs = [detect_pii(text, pii_labels) for text in vanilla_outputs]
        pii_ours_outputs = [detect_pii(text, pii_labels) for text in ours_outputs]

        pii_results = {
            "pii_instructions": pii_instructions,
            "pii_vanilla_outputs": pii_vanilla_outputs,
            "pii_ours_outputs": pii_ours_outputs
        }
        save_pii_results(pii_results, save_file)
    else:
        print("\nLoading Locally")
        pii_instructions = pii_results["pii_instructions"]
        pii_vanilla_outputs = pii_results["pii_vanilla_outputs"]
        pii_ours_outputs = pii_results["pii_ours_outputs"]

    # Analyze PII additions and drops
    vanilla_added_records = []
    vanilla_dropped_records = []
    ours_added_records = []
    ours_dropped_records = []

    for idx, (instr, vanilla_out, ours_out) in enumerate(zip(pii_instructions, pii_vanilla_outputs, pii_ours_outputs)):
        added_vanilla, dropped_vanilla = compute_pii_changes_entailment(instr, vanilla_out) #compute_pii_changes(instr, vanilla_out)
        added_ours, dropped_ours = compute_pii_changes_entailment(instr, ours_out) #compute_pii_changes(instr, ours_out)
        
        for added in added_vanilla:
            vanilla_added_records.append({
                "Index": idx,
                "Original Example": instructions[idx],
                "Original PII Values": [item['text'] for item in instr],
                "vanilla Output": vanilla_outputs[idx],
                "vanilla PII Values": [item['text'] for item in vanilla_out],
                "Added PII": added
            })
        
        for dropped in dropped_vanilla:
            vanilla_dropped_records.append({
                "Index": idx,
                "Original Example": instructions[idx],
                "Original PII Values": [item['text'] for item in instr],
                "vanilla Output": vanilla_outputs[idx],
                "vanilla PII Values": [item['text'] for item in vanilla_out],
                "Dropped PII": dropped
            })
        
        for added in added_ours:
            ours_added_records.append({
                "Index": idx,
                "Original Example": instructions[idx],
                "Original PII Values": [item['text'] for item in instr],
                "Our Output": ours_outputs[idx],
                "Our PII Values": [item['text'] for item in ours_out],
                "Added PII": added
            })
        
        for dropped in dropped_ours:
            ours_dropped_records.append({
                "Index": idx,
                "Original Example": instructions[idx],
                "Original PII Values": [item['text'] for item in instr],
                "Our Output": ours_outputs[idx],
                "Our PII Values": [item['text'] for item in ours_out],
                "Dropped PII": dropped
            })

    # Save added and dropped records to CSV files
    path = "/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/hallucination_results/xsum/vanilla"
    pd.DataFrame(vanilla_added_records).to_csv(f"{path}/vanilla_pii_added_2.csv", index=False)
    pd.DataFrame(vanilla_dropped_records).to_csv(f"{path}/vanilla_pii_dropped_2.csv", index=False)
    pd.DataFrame(ours_added_records).to_csv(f"{path}/ours_pii_added_2.csv", index=False)
    pd.DataFrame(ours_dropped_records).to_csv(f"{path}/ours_pii_dropped_2.csv", index=False)

    # Prepare data for plotting
    freq_data = pd.DataFrame({
        "Method": ["vanilla", "vanilla", "Ours", "Ours"], 
        "PII Type": ["Added", "Dropped", "Added", "Dropped"],
        "Count": [len(vanilla_added_records), len(vanilla_dropped_records), len(ours_added_records), len(ours_dropped_records)]
    })

    # Plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plotting bar chart
    ax = sns.barplot(x="Method", y="Count", hue="PII Type", data=freq_data, palette=['#5DADE2', '#E74C3C'])

    # Annotate bars with actual numbers
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(format(height, 'd'),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    plt.title("PII Additions and Drops")
    plt.ylabel("Count")
    plt.xlabel("Method")
    plt.legend(title="PII Type")
    plt.tight_layout()

    # Save plot as PDF
    plt.savefig(output_file, format='pdf', dpi=400)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PII additions and drops in generated text outputs.")
    parser.add_argument("-d", "--vanilla_file", type=str, required=True, help="Path to the vanilla results JSON file.")
    parser.add_argument("-o", "--ours_file", type=str, required=True, help="Path to our method results JSON file.")
    parser.add_argument("-f", "--output_file", type=str, required=True, help="Path to save the output PDF file.")
    parser.add_argument("-s", "--save_file", type=str, required=True, help="Path to save the PII detection results.")

    args = parser.parse_args()
    main(args.vanilla_file, args.ours_file, args.output_file, args.save_file)
