from networkx.readwrite import json_graph
import json, torch, argparse
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler


def parse_arguments():
    parser = argparse.ArgumentParser('Argument Quality')
    parser.add_argument('--dir_index', type=str, default="1")

    return parser.parse_args()


args = parse_arguments()
print(f"args.dir_index: ", args.dir_index)

save_path = 'roberta-base-finetuned/'

# Load the fine-tuned model and tokenizer
model_tuned = RobertaForSequenceClassification.from_pretrained(save_path + "model")
tokenizer_tuned = RobertaTokenizer.from_pretrained(save_path + "tokenizer")

# If you have a GPU, move the model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_tuned.to(device)

def preprocess_function(examples):
    # Concatenate topic and argument with a separator
    concatenated_inputs = [topic + " [SEP] " + argument for topic, argument in
                           zip(examples['topic'], examples['argument'])]
    return tokenizer_tuned(concatenated_inputs, truncation=True, padding='max_length', max_length=256)


def evaluate(model, eval_dataloader):
    model.eval()
    all_predictions = []
    for batch in eval_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()  # Move the predictions back to CPU and convert to numpy

        all_predictions.extend(predictions)  # Append the predictions for the batch
    print(f"all_predictions: {all_predictions}")
    return all_predictions

# topic = "The impact of climate change on agriculture."
# argument = "Climate change has severely affected crop yields due to unpredictable weather patterns."
#
# score = predict_relevance(topic, argument)
# print(f"Relevance Score: {score}")


# with open("/scratch/rrs99/Stance_Distribution/community_detection_main_classes/output_" + cur_topic + "_allsides_200_articles_discourse_arguments.json", "r") as file_j:

test_df = pd.read_csv('/scratch/rrs99/argmining-21-keypoint-analysis-sharedtask-code/KPA_2021_shared_task/kpm_data/key_points_train.csv')
# the following line is only for key_points_train
test_df = test_df.rename(columns={"key_point": "argument"}, errors="raise")

test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

test_dataloader = DataLoader(test_dataset, batch_size=8)

test_df['quality_score'] = evaluate(model_tuned, test_dataloader)

# the following line is only for key_points_train
test_df = test_df.rename(columns={"argument": "key_point"}, errors="raise")

json_file_path = '/scratch/rrs99/argmining-21-keypoint-analysis-sharedtask-code/KPA_2021_shared_task/kpm_data/key_points_train_with_quality_score.json'
csv_file_path = '/scratch/rrs99/argmining-21-keypoint-analysis-sharedtask-code/KPA_2021_shared_task/kpm_data/key_points_train_with_quality_score.csv'

test_df.to_csv(csv_file_path, index=False)
test_df.to_json(json_file_path, orient='records')
