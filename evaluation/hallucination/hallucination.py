import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from gliner import GLiNER

# Load the GLiNER PII model
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

# Function to detect PII
def detect_pii(text, labels):
    try:
        entities = model.predict_entities(text, labels)
    except KeyError as e:
        print(f"KeyError: {e}")
        entities = []
    return entities

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

# Function to count PII entities
def count_pii_entities(entities):
    return len(entities)

# Function to convert PII entities to a hashable form
def pii_to_hashable(entities):
    return set(tuple(entity.items()) for entity in entities)

# Function to count PII entities that are added or dropped
def compute_pii_changes(original, modified):
    original_set = pii_to_hashable(original)
    modified_set = pii_to_hashable(modified)
    added = modified_set - original_set
    dropped = original_set - modified_set
    return len(added), len(dropped)

# Main function
def main(dpsgd_file, ours_file, output_file, save_file):
    # Load data
    dpsgd_data = load_data(dpsgd_file)
    ours_data = load_data(ours_file)

    # Extract instructions and outputs
    instructions = [item['instruction'] for item in dpsgd_data]
    dpsgd_outputs = [item['output'] for item in dpsgd_data]
    ours_outputs = [item['output'] for item in ours_data]

    # Load or detect PII
    pii_results = load_pii_results(save_file)
    if not pii_results:
        pii_instructions = [detect_pii(text, pii_labels) for text in instructions]
        pii_dpsgd_outputs = [detect_pii(text, pii_labels) for text in dpsgd_outputs]
        pii_ours_outputs = [detect_pii(text, pii_labels) for text in ours_outputs]

        pii_results = {
            "pii_instructions": pii_instructions,
            "pii_dpsgd_outputs": pii_dpsgd_outputs,
            # "pii_dpsgd_outputs": pii_dpsgd_outputs,
            "pii_ours_outputs": pii_ours_outputs
        }
        save_pii_results(pii_results, save_file)
    else:
        pii_instructions = pii_results["pii_instructions"]
        pii_dpsgd_outputs = pii_results["pii_dpsgd_outputs"]
        pii_ours_outputs = pii_results["pii_ours_outputs"]

    # Analyze PII additions and drops
    pii_added_dpsgd, pii_dropped_dpsgd = 0, 0
    pii_added_ours, pii_dropped_ours = 0, 0

    for instr, dpsgd_out, ours_out in zip(pii_instructions, pii_dpsgd_outputs, pii_ours_outputs):
        added_dpsgd, dropped_dpsgd = compute_pii_changes(instr, dpsgd_out)
        added_ours, dropped_ours = compute_pii_changes(instr, ours_out)
        pii_added_dpsgd += added_dpsgd
        pii_dropped_dpsgd += dropped_dpsgd
        pii_added_ours += added_ours
        pii_dropped_ours += dropped_ours

    print(f"pii_added_dpsgd: {pii_added_dpsgd}, \tpii_dropped_dpsgd: {pii_dropped_dpsgd},\npii_added_ours: {pii_added_ours}, \tpii_dropped_ours: {pii_dropped_ours}")

    # Prepare data for plotting
    freq_data = pd.DataFrame({
        "Method": ["DPSGD", "DPSGD", "Ours", "Ours"],
        "PII Type": ["Added", "Dropped", "Added", "Dropped"],
        "Count": [pii_added_dpsgd, pii_dropped_dpsgd, pii_added_ours, pii_dropped_ours]
    })

    # Plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plotting bar chart
    sns.barplot(x="Method", y="Count", hue="PII Type", data=freq_data, palette=['#5DADE2', '#E74C3C'])

    # plt.title("PII Additions and Drops")
    plt.ylabel("Count")
    plt.xlabel("Method")
    plt.legend(title="PII Type")
    plt.tight_layout()

    # Save plot as PDF
    plt.savefig(output_file, format='pdf', dpi=400)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PII additions and drops in generated text outputs.")
    parser.add_argument("-d", "--dpsgd_file", type=str, required=True, help="Path to the DPSGD results JSON file.")
    parser.add_argument("-o", "--ours_file", type=str, required=True, help="Path to our method results JSON file.")
    parser.add_argument("-f", "--output_file", type=str, required=True, help="Path to save the output PDF file.")
    parser.add_argument("-s", "--save_file", type=str, required=True, help="Path to save the PII detection results.")

    args = parser.parse_args()
    main(args.dpsgd_file, args.ours_file, args.output_file, args.save_file)
