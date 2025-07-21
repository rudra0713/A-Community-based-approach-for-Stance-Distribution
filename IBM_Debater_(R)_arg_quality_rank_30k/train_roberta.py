import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

# Load the dataset
csv_file_path = "arg_quality_rank_30k.csv"

df = pd.read_csv(csv_file_path)


# Preprocess the dataset: Concatenate the 'topic' and 'argument' for each example
def preprocess_function(examples):
    # Concatenate topic and argument with a separator
    concatenated_inputs = [topic + " [SEP] " + argument for topic, argument in
                           zip(examples['topic'], examples['argument'])]
    return tokenizer(concatenated_inputs, truncation=True, padding='max_length', max_length=256)


# Split dataset into train, eval, and test
train_df = df[df['set'] == 'train']
eval_df = df[df['set'] == 'dev']
test_df = df[df['set'] == 'test']

# Convert to Huggingface dataset format
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)  # Regression task

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set input format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'WA'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'WA'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'WA'])

# Define DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training and evaluation loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def evaluate(model, eval_dataloader):
    model.eval()
    mse_loss_fn = torch.nn.MSELoss()
    total_loss = 0

    for batch in eval_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
        labels = batch['WA'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze()
            loss = mse_loss_fn(predictions, labels)
            total_loss += loss.item()

    return total_loss / len(eval_dataloader)

epochs = 3
mse_loss_fn = torch.nn.MSELoss()

for epoch in range(epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
        labels = batch['WA'].to(device)

        outputs = model(**inputs)
        predictions = outputs.logits.squeeze()

        loss = mse_loss_fn(predictions, labels)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Evaluation
    eval_loss = evaluate(model, eval_dataloader)
    print(f"Epoch {epoch + 1}: Eval Loss: {eval_loss}")

# Testing
test_loss = evaluate(model, test_dataloader)
print(f"Test Loss: {test_loss}")


save_path = 'roberta-base-finetuned/'
model.save_pretrained(save_path + "model")
tokenizer.save_pretrained(save_path + "tokenizer")


# Load the fine-tuned model and tokenizer
model_tuned = RobertaForSequenceClassification.from_pretrained(save_path + "model")
tokenizer_tuned = RobertaTokenizer.from_pretrained(save_path + "tokenizer")

# Set the model to evaluation mode
model_tuned.eval()

# If you have a GPU, move the model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_tuned.to(device)


def predict_relevance(topic, argument):
    # Concatenate topic and argument with separator
    input_text = topic + " [SEP] " + argument

    # Tokenize the input
    inputs = tokenizer_tuned(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)

    # Move the inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Predict with the model
    with torch.no_grad():
        outputs = model_tuned(**inputs)
        # Get the predicted score
        score = outputs.logits.squeeze().item()

    return score


topic = "The impact of climate change on agriculture."
argument = "Climate change has severely affected crop yields due to unpredictable weather patterns."

score = predict_relevance(topic, argument)
print(f"Relevance Score: {score}")