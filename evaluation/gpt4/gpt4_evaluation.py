import openai
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm 

OPENAI_API_KEY = ''
# export OPENAI_API_KEY=

def evaluate_summary(option1, option2, model="gpt-4-0125-preview"):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # prompt = f"Which of the following summaries is better? (Choose Option 1 or Option2, don't provide a justification)\nOption1: \"{option1}\"\nOption2: \"{option2}\""
    # prompt = f"Please read the following two summaries carefully and choose which one is better in terms of clarity, coherence, and information content. (Choose Option 1 or Option2, don't provide a justification).\n\nOption 1:\n\"{option1}\"\n\nOption 2:\n\"{option2}\"\n\nChoice:"

    # prompt = f"""
    #             Please carefully evaluate the following two summaries based on the criteria of clarity, coherence, and information content. After reading both, select the option that you find superior in meeting these criteria. Your choice should be concise, either 'Option 1' or 'Option 2', without providing any justification.

    #             - Clarity: How easily understandable and clear is the summary?
    #             - Coherence: How well do the ideas flow together?
    #             - Information Content: How effectively does the summary convey the essential information of the original text?

    #             Option 1:
    #             "{option1}"

    #             Option 2:
    #             "{option2}"

    #             Your choice (Option 1 or Option 2 only):
    #             """

    prompt = f""" Please read the following two summaries carefully. Evaluate them based on clarity, coherence, information content, conciseness, accuracy, relevance, objectivity, flow and structure, engagement, novelty, and grammar and style. Choose which one is better overall based on these criteria. (Choose Option 1 or Option 2, without providing a justification).
                Option 1:
                "{option1}"

                Option 2:
                "{option2}"

                Choice:
            """

    response = client.chat.completions.create(
        model=model, 
        messages=[
            {"role": "system", "content": "You are a highly intelligent AI trained to evaluate text summaries."},
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content.strip()
    # Extract the preference based on the response
    return 1 if "Option 1" in text else 2 if "Option 2" in text else 0

# def process_csv_and_flip_options(file_path, output_file_path):
#     # df = pd.read_csv(file_path)
#     df = pd.read_csv(file_path, encoding='utf-8')
    
#     original_order_results = []
#     flipped_order_results = []

#     # Wrap the main loop with tqdm for a progress bar
#     for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Summaries"):
#         # if index > 1:
#         #     break
#         original_scores = []
#         flipped_scores = []
#         for _ in range(5):  # Repeat 5 times for each row in original and flipped order
#             original_score = evaluate_summary(row['Option 1'], row['Option 2'])
#             original_scores.append(original_score)
            
#             flipped_score = evaluate_summary(row['Option 2'], row['Option 1'])
#             flipped_scores.append(3 - flipped_score)  # Adjust the score based on flipped options
        
#         original_mean_score = np.mean(original_scores)
#         flipped_mean_score = np.mean(flipped_scores)

#         original_order_results.append(original_mean_score)
#         flipped_order_results.append(flipped_mean_score)
    
#     df['Original Mean Score'] = original_order_results
#     df['Flipped Mean Score'] = flipped_order_results

#     df.to_csv(output_file_path, index=False)
#     return df

def process_csv_and_flip_options(file_path, output_file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    
    results = [] 

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Summaries"):
        original_scores = []
        flipped_scores = []
        for _ in range(5):  # Repeat 5 times for each row in original and flipped order
            original_score = evaluate_summary(row['Option 1'], row['Option 2'])
            original_scores.append(original_score)
            
            flipped_score = evaluate_summary(row['Option 2'], row['Option 1'])
            flipped_scores.append(3 - flipped_score)  # Adjust the score based on flipped options
        
        original_mean_score = np.mean(original_scores)
        flipped_mean_score = np.mean(flipped_scores)

        results.append([index, original_mean_score, flipped_mean_score])
    
    results_df = pd.DataFrame(results, columns=['Index', 'Original Mean Score', 'Flipped Mean Score'])
    results_df.to_csv(output_file_path, index=False)
    
    return results_df

dir = "/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/evaluation"
file_path = f'{dir}/results_200.csv'
output_file_path = f'{dir}/gpt4_evaluation_4.csv' 
result_df = process_csv_and_flip_options(file_path, output_file_path)

# Calculate some basic statistics
original_mean = result_df['Original Mean Score'].mean()
flipped_mean = result_df['Flipped Mean Score'].mean()

# Count how many times Option 1 or Option 2 was preferred in original and flipped scenarios
original_preference = result_df['Original Mean Score'].value_counts().to_dict()
flipped_preference = result_df['Flipped Mean Score'].value_counts().to_dict()

print("Overall Insights:")
print(f"Average score for original options: {original_mean:.2f}")
print(f"Average score for flipped options: {flipped_mean:.2f}\n")

print("Preference Count in Original Order:")
for score, count in original_preference.items():
    print(f"Score {score}: {count} times")

print("\nPreference Count in Flipped Order:")
for score, count in flipped_preference.items():
    print(f"Score {score}: {count} times")

# Additional insights: Compare if flipping the options changes preference significantly
changes_in_preference = (result_df['Original Mean Score'] != result_df['Flipped Mean Score']).sum()
total_rows = len(result_df)
print(f"\nNumber of times flipping the options changed the preference: {changes_in_preference} out of {total_rows} comparisons.")


# Option1: DPSGD_Predicted_Summary
# Option2: ImpossibleDistillation_Predicted_Summary