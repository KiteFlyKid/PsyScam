import pandas as pd
from openai import OpenAI

import json
import os
import argparse

parser = argparse.ArgumentParser(description="Run LLM-based prediction script.")
parser.add_argument("--csv", type=str, required=True, help="Name or path of the csv")
parser.add_argument("--col", type=str, required=True, help="column name in the csv to evaluate")
args = parser.parse_args()

csv_name = args.csv
col = args.col

df = pd.read_csv(csv_name)  # final label file for bbb phishing

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




principles_file_path = "data/human_labeled/PTs.csv"
principles_df = pd.read_csv(principles_file_path)
principle_table = "\n".join(
    [
        f"- **{row['Technique'].strip()}**: {row['Definition']} "
        # f"(Example: {row['Example']})"
        for _, row in principles_df.iterrows()
    ]
)

Background_prompt = f"""
Scammers use social engineering attacks that exploit psychological techniques to manipulate victims.

Our goal: We collect a number of scam reports and aim to extract the psychological techniques used in these scam reports.

"""

task_prompts = {'report_idx': [], 'prompt': []}
for idx, row in df.iterrows():
    scam_desc = row[col]

    task_prompt = [
        {"role": "system", "content": Background_prompt},  # Background information
        {"role": "user", "content": f"""Now I give you the victim report: {scam_desc}
                Please extract any psychological techniques exploited by the scammer.
             We consider the 9 psychological techniques:
{principle_table}
                Requirement 1: if no psychological techniques is identified, return an empty dictionary:{{}}. 
             Avoid guess, you must return the psychological techniques when you are prettry sure it exists 
                Requirement 2: Return the output strictly as a JSON dictionary:
                {{
                    "psychological technique A": Corresponding texts in the victim report,
                    "psychological technique B": Corresponding texts in the victim report,
                    ...
                }}
                For example:
                {{'Reciprocity': "This is a work from home job which means that you‘ll have the opportunity to operate and personally plan your own day schedule while remaining in the safety of your home atmosphere.", 
                'Consistency': "We received your application for our Remote Customer Enrollment Position. Our hiring manager reviewed your resume and would love to set up an interview with you via zoom."}}    
            """}
    ]

    task_prompts['report_idx'].append(idx)
    task_prompts['prompt'].append(task_prompt)



backup_file = f"data/LLM_labeled/{csv_name}.json"
output_file = f"data/evaluated/{csv_name}_gpt41extracted.csv"
failed_responses_file = f"data/LLM_labeled/{csv_name}_failed_responses.json"

# Ensure directories exist
os.makedirs(os.path.dirname(backup_file), exist_ok=True)

# Load existing responses (incremental saving)
if os.path.exists(backup_file):
    with open(backup_file, "r") as f:
        responses = json.load(f)
else:
    responses = []

failed_response = []

# Resume processing from last saved response
start_index = len(responses)
print(f"Resuming from index: {start_index}")
for i, task_prompt in enumerate(task_prompts['prompt'][start_index:], start=start_index):
    try:
        response = ask(task_prompt, token=4 * 4096, temp=1, model="gpt41")
        response_text = response.split('</think>\n')[-1].strip()
        responses.append(response_text)

        # Save incremental responses
        with open(backup_file, "w") as f:
            json.dump(responses, f, indent=4)

    except Exception as e:
        print(f"Error processing prompt {i}: {e}")
        responses.append(None)
        failed_response.append({"index": i, "prompt": task_prompt, "error": str(e)})

# Final DataFrame creation
response_df = df.copy()
response_df['LLM_evaluator'] = responses


# JSON extraction function
def extract_json(text):
    try:
        cleaned_text = text.strip().replace("```json\n", "").replace("\n```", "").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        failed_response.append({"raw_text": text, "error": str(e)})
        return None


# Apply JSON extraction
response_df["LLM_evaluator"] = response_df["LLM_evaluator"].apply(extract_json)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)
response_df.to_csv(output_file, index=False)

# Log failed responses
if failed_response:
    with open(failed_responses_file, "w") as f:
        json.dump(failed_response, f, indent=4)

print("Processing complete. Responses saved!")