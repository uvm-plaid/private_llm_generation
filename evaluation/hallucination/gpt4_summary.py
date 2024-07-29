import pandas as pd
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from openai import OpenAI

# Load dataset
df = pd.read_csv('./data_1000.csv')

# Initialize OpenAI GPT (assuming you have access to GPT-4 through OpenAI's API)
# openai.api_key = 'your_openai_api_key'
OPENAI_API_KEY = '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'

# export OPENAI_API_KEY=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

# Example function to generate summary with OpenAI's GPT
def generate_summary_gpt4(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    full_prompt = "Summarize the following text: " + text

    messages = [
            {"role": "system", "content": "You are a highly intelligent AI trained to evaluate text summaries."},
            {"role": "user", "content": full_prompt}
        ]

    response = client.chat.completions.create(
      model="gpt-4-0125-preview", #"text-davinci-003",  # Replace with GPT-4 engine when available
      messages=messages,
      temperature=0.7,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    # return response.choices[0].text.strip()
    return response.choices[0].message.content.strip()

# Apply the function to each row in the dataframe
df['Generated Summary'] = df['Original Text'].apply(generate_summary_gpt4)

# Example to save the results back to a new CSV
df.to_csv('./data_gen_1000.csv', index=False)
