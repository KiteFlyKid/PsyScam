import pandas as pd
import numpy as np
import ast
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Run LLM-based prediction script.")
parser.add_argument("--llm_model", type=str, required=True, help="Name or path of the LLM model to use")
parser.add_argument("--num_added_pts", type=int, default=1, help="Number of new persuasion techniques to add")

args = parser.parse_args()

llm_model = args.llm_model
num_added_pts = args.num_added_pts

print(f"Using LLM model: {llm_model}")
print(f"Adding {num_added_pts} new persuasion technique(s)")

scam_df = pd.read_csv('data/D2.csv')
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
# scam_df = scam_df[scam_df['all_pts'].apply(lambda x: len(x.split(',')) == 2)]
# scam_df=scam_df[scam_df['all_pts'].apply(lambda x: len(x) == 2)]
# print(f"Filtered to {len(scam_df)} rows with exactly 2 PTs")
pts_df = pd.read_csv('data/PTs.csv')
# Create mapping from PT label to definition
pt_label_to_def = dict(zip(pts_df['Label'], pts_df['Definition']))
pt_label_to_name = dict(zip(pts_df['Label'], pts_df['Technique']))


# Modified function to select multiple new PTs that aren't already in the message
def select_new_pts(current_pts, num_to_add):
    available_pts = [pt for pt in all_pt_labels if pt not in current_pts]

    # If we have enough available PTs, randomly select the requested number
    if len(available_pts) >= num_to_add:
        return random.sample(available_pts, num_to_add)
    # If we don't have enough available PTs, return as many as we can
    elif available_pts:
        return available_pts
    # If no PTs are available, return None
    return None


augmented_df = scam_df.copy()
all_pt_labels = list(pt_label_to_def.keys())

# Add the new_pts column
augmented_df['parsed_pts'] = scam_df['all_pts'].apply(lambda x: x.split(','))
augmented_df['new_pts'] = augmented_df['parsed_pts'].apply(lambda x: select_new_pts(x, num_added_pts))

# Drop rows where no new PTs could be added
augmented_df = augmented_df.dropna(subset=['new_pts'])
print(f"Retained {len(augmented_df)} rows after filtering for available new PTs")


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



augmented_df['rewritten_msg'] = None

from openai import OpenAI
import time

responses = []
for idx in tqdm(range(len(augmented_df))):
    # Get the current row
    row = augmented_df.iloc[idx]

    # Get the message and PT information
    message = row['description']
    current_pts = row['parsed_pts']
    new_pts_list = row['new_pts']

    # Format the new PTs for the prompt
    new_pts_details = []
    for pt in new_pts_list:
        pt_name = pt_label_to_name.get(pt, "Unknown")
        pt_def = pt_label_to_def.get(pt, "Unknown")
        new_pts_details.append(f"{pt} - {pt_name}: {pt_def}")

    # Create the prompt for the model
    prompt = f"""
    You are helping to rewrite a scam message to include {len(new_pts_list)} new psychological technique(s) while keeping the original facts the same.

    Original message:
    "{message}"

    Current psychological techniques used: {', '.join([pt_label_to_name.get(current_pt) for current_pt in current_pts])}

    Please rewrite this message to also include the following psychological technique(s):
    {' '.join(new_pts_details)}

    Make sure to keep all the original facts intact while incorporating these new techniques. 
    Only return the rewritten message without any additional text or explanation.
    """

    try:
        message = [
            {"role": "system",
             "content": "You are an assistant that rewrites scam messages incorporating specific psychological techniques"},
            {"role": "user", "content": prompt}
        ]
        # Call the OpenAI API
        if 'qwen' in llm_model or 'llama' in llm_model:
            response = ask(message, token=4 * 2048, temp=0.7, model='local')
        else:
            response = ask(message, token=4 * 2048, temp=0.7, model=llm_model)

        if response:
            # Clean the response (remove quotes if present)
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            responses.append(response)
        else:
            responses.append(None)

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        if len(responses) <= idx:
            responses.append(None)
        # Add a longer delay if hitting rate limits
        time.sleep(3)

with open(f'responses_{llm_model}_{num_added_pts}pts.txt', 'w') as f:
    for response in responses:
        if response:
            f.write(response + '\n')
        else:
            f.write("ERROR: No response\n")

augmented_df['rewritten_msg'] = responses

# Convert the list of new PTs to a comma-separated string for easier handling
augmented_df['new_pts_str'] = augmented_df['new_pts'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')

# Save result with number of added PTs in filename
augmented_df.to_csv(f'data/augmentation_{llm_model}_{num_added_pts}pts.csv', index=False)

print(f"Completed! Added {num_added_pts} PT(s) to {len(augmented_df)} messages using {llm_model}.")
print(f"Results saved to data/augmentation_{llm_model}_{num_added_pts}pts.csv")