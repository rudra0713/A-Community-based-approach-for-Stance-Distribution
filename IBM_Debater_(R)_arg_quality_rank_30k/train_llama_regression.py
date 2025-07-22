from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from huggingface_hub import login
import numpy as np
import torch, json
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

model_name = 'meta-llama/Meta-Llama-3-8B'
output_dir = "llama-3.1-fine-tuned-model-regression"
login(token=os.getenv("HF_TOKEN"))
access_token = os.getenv("HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLaMAForRegression(nn.Module):
    def __init__(self, llama_model):
        super(LLaMAForRegression, self).__init__()
        self.llama = llama_model
        # Replace the final classification head with a regression head
        self.regression_head = nn.Linear(llama_model.config.hidden_size, 1)  # Output single continuous value

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Move inputs to the correct device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass through LLaMA model
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        # Apply regression head
        prediction = self.regression_head(last_hidden_state[:, -1, :])
        return prediction


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config,
    token=access_token,
)

# Ensure model is on the correct device
model = LLaMAForRegression(model).to(device)

# For multi-GPU support, wrap with DataParallel (if not using DDP)
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Use the appropriate devices

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
tokenizer.pad_token_id = tokenizer.eos_token_id

csv_file_path = "arg_quality_rank_30k.csv"
df = pd.read_csv(csv_file_path)
# df['WA'] = df['WA'].apply(lambda x: 'relevant' if x >= 0.5 else 'irrelevant')

# Create three new DataFrames based on the "set" column values
X_train = df[df['set'] == 'train']
X_eval = df[df['set'] == 'dev']
X_test = df[df['set'] == 'test']
X_train = X_train.head(200)
X_eval = X_eval.head(200)
X_test = X_test.head(100)


# Define the prompt generation functions
def generate_prompt(data_point):
    return f"""
            Predict the quality score of the argument with respect to the topic.
            argument: {data_point["argument"]}
            topic: {data_point["topic"]}            
            label: {data_point["WA"]}""".strip()


def generate_test_prompt(data_point):
    return f"""
            Predict the quality score of the argument with respect to the topic.
            argument: {data_point["argument"]}
            topic: {data_point["topic"]}            
            label: """.strip()


# Generate prompts for training and evaluation data
X_train.loc[:, 'text'] = X_train.apply(generate_prompt, axis=1)
X_eval.loc[:, 'text'] = X_eval.apply(generate_prompt, axis=1)

# Generate test prompts and extract true labels
y_true = X_test.loc[:, 'WA']
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

train_data = Dataset.from_pandas(X_train[["text"]])
eval_data = Dataset.from_pandas(X_eval[["text"]])


def tokenize_function(data):
    return tokenizer(data['text'], truncation=True, padding='max_length', max_length=512)


train_data = train_data.map(tokenize_function, batched=True)
eval_data = eval_data.map(tokenize_function, batched=True)

# Set the format for PyTorch
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
eval_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    return {"mse": ((predictions - labels) ** 2).mean()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    compute_metrics=compute_metrics,
)

trainer.train()
