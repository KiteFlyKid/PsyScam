import pandas as pd
import numpy as np
import ast
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from openai import OpenAI
import re
import time

import argparse

parser = argparse.ArgumentParser(description="Run LLM-based prediction script.")
parser.add_argument("--llm_model", type=str, required=True, help="Name or path of the LLM model to use")

args = parser.parse_args()

llm_model = args.llm_model


def ask(prompt, token, temp, model):
    with open('api.key', 'r') as f:
        api_key = f.read()
    with open('grok.key', 'r') as f:
        grok_key = f.read()
    with open('gemini.key', 'r') as f:
        gemini_key = f.read()
    # 设置API基础URL
    if 'gpt' in model:
        api_base = 'https://api.openai.com/v1'
    elif 'local' in model:
        api_base = "http://localhost:8000/v1"
    elif 'grok' in model:
        api_base = "https://api.x.ai/v1"
        api_key = grok_key
    elif 'gemini' in model:
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = gemini_key

    if model == 'gemini':
        setmodel = 'gemini-2.0-flash'
    if model == 'gpt41mini':
        setmodel = 'gpt-4.1-mini-2025-04-14'
    if model == 'grok3':
        setmodel = 'grok-3-latest'
    if model == 'grok3mini':
        setmodel = 'grok-3-mini'
    if model == 'gpt41':
        setmodel = 'gpt-4.1-2025-04-14'
    if model == 'gpto4mini':
        setmodel = 'o4-mini-2025-04-16'
    if model == "gpto1":
        setmodel = 'o1-2024-12-17'
    if model == "gpto1mini":
        setmodel = "o1-mini-2024-09-12"
    if model == "gpt4o":
        setmodel = 'gpt-4o-2024-11-20'
    if model == "gpt4omini":
        setmodel = "gpt-4o-mini-2024-07-18"
    if model == "gpt4t":
        setmodel = "gpt-4-turbo"
    else:
        client = OpenAI(api_key=api_key, base_url=api_base)

    if "o1" not in setmodel and "o3" not in setmodel and "o4" not in setmodel:
            stream = client.chat.completions.create(
                model=setmodel,
                messages=prompt,
                stream=True,
                max_tokens=token,
                temperature=temp,
            )
            final_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    final_response += chunk.choices[0].delta.content
    else:
            final_response = client.chat.completions.create(
                model=setmodel,
                messages=prompt,
            )
            final_response = final_response.choices[0].message.content

    return final_response


scam_df = pd.read_csv('data/D2.csv')  # final label file for bbb phishing
scam_df = scam_df.sample(frac=0.3, random_state=42).reset_index(drop=True)
pts_df = pd.read_csv('data/PTs.csv')
pt_label_mapping = {
    "Authority and Impersonation": "a",
    "Phantom Riches": "p",
    "Fear and Intimidation": "f",
    "Liking": "l",
    "Urgency and Scarcity": "s",
    "Pretext and trust": "t",
    "Reciprocity": "r",
    "Consistency": "c",
    "Social Proof": "sp"
}


# Helper function to map PT names to labels
def map_pts_to_labels(pt_string):
    if pd.isna(pt_string):
        return ""
    try:
        pt_dict = eval(pt_string)
        return ",".join([pt_label_mapping[pt] for pt in pt_dict.keys() if pt in pt_label_mapping])
    except Exception:
        return ""


scam_df['all_pts'] = scam_df['PTs'].apply(map_pts_to_labels)
scam_df = scam_df[scam_df['all_pts'].apply(lambda x: len(x) > 0)]
autocomplete_df = pd.DataFrame()

# Define split percentages
split_percentages = [0.2, 0.4, 0.6]


# Function to split text at a given percentage of words
def split_text_at_percentage(text, percentage):
    words = word_tokenize(text)
    split_point = int(len(words) * percentage)
    beginning = ' '.join(words[:split_point])
    ending = ' '.join(words[split_point:])
    return beginning, ending


# Create multiple entries for each text with different split percentages
rows = []
for idx, row in scam_df.iterrows():
    text = row['description']
    pts = row['all_pts']

    # Skip texts that are too short (less than 20 words)
    if len(word_tokenize(text)) < 20:
        continue

    for percentage in split_percentages:
        beginning, ending = split_text_at_percentage(text, percentage)

        rows.append({
            'original_id': row['scam_id'],
            'full_text': text,
            'beginning': beginning,
            'ending': ending,
            'split_percentage': percentage,
            'pts': pts
        })

# Create the dataframe
autocomplete_df = pd.DataFrame(rows)

# Display information about the dataset
print(f"Created {len(autocomplete_df)} samples for autocomplete task")
print(f"Split percentages used: {split_percentages}")
print("\nExample splits:")
for percentage in split_percentages:
    example = autocomplete_df[autocomplete_df['split_percentage'] == percentage].iloc[0]
    print(f"\nSplit at {percentage * 100}%:")
    print(f"Beginning: {example['beginning'][:100]}...")
    print(f"Ending: {example['ending'][:100]}...")

llm_df = autocomplete_df
llm_df['generated_completion'] = None
llm_df['full_generated_text'] = None

# Create mappings from PT label to definition and name
pt_label_to_def = dict(zip(pts_df['Label'], pts_df['Definition']))
pt_label_to_name = dict(zip(pts_df['Label'], pts_df['Technique']))


# Function to format psychological techniques with their definitions
def format_pts_with_definitions(pts):
    formatted_pts = []
    for pt in pts:
        pt_name = pt_label_to_name.get(pt, "Unknown")
        pt_def = pt_label_to_def.get(pt, "Unknown")
        formatted_pts.append(f"{pt} - {pt_name}: {pt_def}")
    return formatted_pts

    # Generate completions for the test set


responses = []

for idx in tqdm(range(len(llm_df))):
    # Get the current row data
    row = llm_df.iloc[idx]
    beginning = row['beginning']
    pts = row['pts']

    # Format PTs with their definitions
    formatted_pts = format_pts_with_definitions(pts)

    # Create the prompt for the model
    prompt = f"""
        You are tasked with completing a scam message based on its beginning. The message should incorporate specific psychological techniques.

        Beginning of the message:
        "{beginning}"

        Please complete this message using the following psychological techniques:
        {chr(10).join(formatted_pts)}

        Ensure your completion continues directly from the last word of the provided beginning, maintaining the same style and tone.
        Only return the completion without any additional text or explanation.
        """

    try:
        message = [
            {"role": "system",
             "content": "You are an assistant that completes scam messages incorporating specific psychological techniques."},
            {"role": "user", "content": prompt}
        ]

        # Call the API
        if 'qwen' in llm_model or 'llama' in llm_model:
            response = ask(message, token=4 * 2048, temp=0.7, model='local')
        else:
            response = ask(message, token=4 * 2048, temp=0.7, model=llm_model)

        # Store the response
        if response:
            # Clean the response (remove quotes if present)
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            # Add to responses list
            responses.append(response)
        else:
            responses.append(None)

        time.sleep(0.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        if len(responses) > idx:
            responses.append(None)
        # Add a longer delay if hitting rate limits
        time.sleep(3)

with open('responses.txt', 'w') as f:
    for response in responses:
        f.write(response + '\n')

if len(responses) < len(llm_df):
    responses += [None] * (len(llm_df) - len(responses))
llm_df['generated_completion'] = responses
llm_df['full_generated_text'] = llm_df['beginning'] + ' ' + llm_df['generated_completion']
llm_df.to_csv(f'data/completion_{llm_model}.csv', index=False)


