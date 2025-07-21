import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from huggingface_hub import login
import numpy as np
import torch, json
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from torch import nn
from networkx.readwrite import json_graph


login(token='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')
access_token = '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'

csv_file_path = "arg_quality_rank_30k.csv"
df = pd.read_csv(csv_file_path)

# Create three new DataFrames based on the "set" column values
X_train = df[df['set'] == 'train']
X_eval = df[df['set'] == 'dev']
X_test = df[df['set'] == 'test']
X_train = X_train.head(200)
X_eval = X_eval.head(50)
X_test = X_test.head(50)


# Define the prompt generation functions
def generate_prompt(data_point):
    return f"""
            Predict the quality score of the argument with respect to the topic on a scale of 0 to 1. Higher score means 
            the argument is highly relevant for the given topic.
            argument: {data_point["argument"]}
            topic: {data_point["topic"]}            
            label: {data_point["WA"]}""".strip()


def generate_test_prompt(data_point):
    return f"""
            Predict the quality score of the argument with respect to the topic on a scale of 0 to 1. Higher score means 
            the argument is highly relevant for the given topic.
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

# base_model_name = "/kaggle/input/llama-3.1/transformers/8b-instruct/1"
#
model_name = 'meta-llama/Meta-Llama-3-8B'
output_dir = "llama-3.1-fine-tuned-model-regression"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
    # torch_dtype="float16",
    # quantization_config=bnb_config,
    token=access_token,
)

model.config.use_cache = False
model.config.pretraining_tp = 1
#
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, num_labels=3)
# model = model.to(device).bfloat16()
tokenizer.pad_token_id = tokenizer.eos_token_id


class LLaMAForRegression(nn.Module):
    def __init__(self, llama_model):
        super(LLaMAForRegression, self).__init__()
        self.llama = llama_model.to(device)

        self.config = llama_model.config

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

    @property
    def device(self):
        return next(self.parameters()).device  # Return the device the model is on


model = LLaMAForRegression(model)


def predict(test, model, tokenizer):
    y_pred = []
    categories = ["relevant", "irrelevant"]

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate text
        outputs = model.llama.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=2,
            temperature=0.1
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("label:")[-1].strip()
        y_pred.append(answer)
        # pipe = pipeline(task="text-generation",
        #                 model=model,
        #                 tokenizer=tokenizer,
        #                 max_new_tokens=2,
        #                 temperature=0.1)
        #
        # result = pipe(prompt)
        # answer = result[0]['generated_text'].split("label:")[-1].strip()
        # for category in categories:
        #     if category.lower() == answer.lower():
        #         y_pred.append(category)
        #         break
        # else:
        #     print(f"unknown answer: {answer}")
        #     y_pred.append("none")
    print(f"final y pred: ")
    print(y_pred)
    return y_pred


def evaluate(y_true, y_pred):
    labels = ["relevant", "irrelevant"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels,
                                         labels=list(range(len(labels))))
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:')
    print(conf_matrix)


print("without fine-tuning model")
y_pred = predict(X_test, model, tokenizer)
# evaluate(y_true, y_pred)
#
#
# def find_all_linear_names(model):
#     cls = bnb.nn.Linear4bit
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#     if 'lm_head' in lora_module_names:  # needed for 16 bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)
#
#
# modules = find_all_linear_names(model)
# print(f"modules: {modules}")
#
# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0,
#     r=64,
#     bias="none",
#     task_type="SEQ_CLS",  # https://discuss.huggingface.co/t/fine-tuning-llm-for-regression-yields-low-loss-during-training-but-not-in-inference/66538/3
#     target_modules=modules,
# )
#
# training_arguments = TrainingArguments(
#     output_dir=output_dir,                    # directory to save and repository id
#     num_train_epochs=3,                       # number of training epochs
#     per_device_train_batch_size=1,            # batch size per device during training
#     gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
#     gradient_checkpointing=True,              # use gradient checkpointing to save memory
#     optim="paged_adamw_32bit",
#     logging_steps=1,
#     learning_rate=2e-4,                       # learning rate, based on QLoRA paper
#     weight_decay=0.001,
#     fp16=True,
#     bf16=False,
#     max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
#     max_steps=-1,
#     warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
#     group_by_length=False,
#     lr_scheduler_type="cosine",               # use cosine learning rate scheduler
#     # report_to="wandb",                  # report metrics to w&b
#     eval_strategy="steps",              # save checkpoint every epoch
#     eval_steps = 0.2
# )
#
# trainer = SFTTrainer(
#     model=model,
#     args=training_arguments,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     max_seq_length=512,
#     packing=False,
#     dataset_kwargs={
#     "add_special_tokens": False,
#     "append_concat_token": False,
#     }
# )
#
# trainer.train()

# print("after fine-tuning")
# y_pred = predict(X_test, model, tokenizer)
# evaluate(y_true, y_pred)
#
# trainer.model.save_pretrained(output_dir)
# model_dir = output_dir + '/' + "model_dir"
#
# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)
#
