import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

df=pd.read_csv(dataset_file)
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



import argparse

parser = argparse.ArgumentParser(description="Run BERT-based Classification script.")
parser.add_argument("--csv", type=str, required=True, help="Name or path of the csv")
args = parser.parse_args()

csv_name = args.csv


# Define the list of possible labels for PTs
labels_list=['a', 'p', 'f', 'l', 's', 't', 'r', 'c', 'sp']


df=pd.read_csv(csv_name)
# Function to process the PTs column into a binary vector
def process_pts(x):
    # Treat "nan" (string) as missing -> no labels
    if x == []:
        parts = []
    else:
        # Split by comma and remove extra spaces; only keep valid labels
        parts = [p.strip() for p in x if p.strip() in labels_list]
    # Create a binary vector for the 6 labels
    return [1 if label in parts else 0 for label in labels_list]

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

# For later conversion of binary vectors back to comma-separated string labels
def vector_to_labels(vector):
    return ",".join([labels_list[i] for i, val in enumerate(vector) if val==1]) or "None"



df['description'] = df['description'].astype(str)

# Process the 'PTs' column to create a multilabel binary vector
df['all_pts'] = df['PTs'].apply(map_pts_to_labels)
df['label_vector'] = df['all_pts'].apply(process_pts)

# (Optional) Create a new column with comma-separated labels for display
df['labels_str'] = df['label_vector'].apply(vector_to_labels)

# Show label distribution: count occurrences for each label across all rows
labels_df = pd.DataFrame(df['label_vector'].tolist(), columns=labels_list)
print("Label distribution (counts):")




threshold = 0.5

Y = np.array(df['label_vector'].tolist())

# Use MultilabelStratifiedShuffleSplit for a stratified train-test split
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in msss.split(df, Y):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

# Rename the column for consistency and convert the values to float
train_df = train_df.rename(columns={"label_vector": "labels"})
test_df = test_df.rename(columns={"label_vector": "labels"})

# Convert label vectors from int to float (for BCEWithLogitsLoss)
train_df['labels'] = train_df['labels'].apply(lambda x: [float(v) for v in x])
test_df['labels'] = test_df['labels'].apply(lambda x: [float(v) for v in x])




# Convert the 'labels' column into a DataFrame for each set
train_labels = pd.DataFrame(train_df["labels"].tolist(), columns=labels_list)
test_labels = pd.DataFrame(test_df["labels"].tolist(), columns=labels_list)
print("Train label distribution (counts):")
print(train_labels.sum())
print("\nTest label distribution (counts):")
print(test_labels.sum())



# Convert the pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Tokenization and Model Training
# ------------------------------------
# Load the Roberta tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["description"], padding="max_length", truncation=True, max_length=512)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns, but keep "description" for later analysis and "labels"
cols_to_remove = set(train_dataset.column_names) - {"input_ids", "attention_mask", "labels", "description"}
train_dataset = train_dataset.remove_columns(list(cols_to_remove))
test_dataset = test_dataset.remove_columns(list(cols_to_remove))

# Set the dataset format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load the pre-trained Roberta model with a classification head for multilabel classification
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=9)
# Set the problem type to multilabel classification so that the correct loss (BCEWithLogitsLoss) is used
model.config.problem_type = "multi_label_classification"

# Define a compute_metrics function for multilabel classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(logits))
    # Threshold probabilities at 0.5 to obtain binary predictions
    predictions = (probs > threshold).int().numpy()
    # Compute subset accuracy (exact match)
    subset_accuracy = np.mean(np.all(predictions == labels, axis=1))
    # Compute micro-averaged precision, recall, and f1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro', zero_division=0)
    return {"subset_accuracy": subset_accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert/results_multilabel",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir='bert/logs_multilabel',
    save_strategy="epoch"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate()
print("Evaluation results:")
print(eval_results)


