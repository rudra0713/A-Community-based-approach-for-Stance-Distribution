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


# topic_list = ['abortion', 'elections', 'gun_control_and_gun_rights', 'immigration', 'taxes', 'voting']
topic_list = ['politics', 'elections', 'gun_control_and_gun_rights', 'immigration', 'us_congress',
              'foreign_policy', 'healthcare', 'environment', 'terrorism', 'education', 'supreme_court',
              'national_security']

cur_topic = topic_list[int(args.dir_index) - 1]
# topic = None
topic = cur_topic.replace('_', ' ')
# if cur_topic == 'abortion':
#     topic = 'abortion'
# elif cur_topic == 'elections':
#     topic = '2024 presidential Election'
# elif cur_topic == 'gun_control_and_gun_rights':
#     topic = 'Gun Control and Gun Rights'
# elif cur_topic == 'immigration':
#     topic = 'Immigration'
# elif cur_topic == 'taxes':
#     topic = 'Taxes'
# elif cur_topic == 'voting':
#     topic = 'Voting'

print(f"focusing on topic: {topic}")

save_path = 'roberta-base-finetuned/'

# Load the fine-tuned model and tokenizer
model_tuned = RobertaForSequenceClassification.from_pretrained(save_path + "model")
tokenizer_tuned = RobertaTokenizer.from_pretrained(save_path + "tokenizer")

# If you have a GPU, move the model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_tuned.to(device)


# def predict_relevance(topic, argument):
#     # Concatenate topic and argument with separator
#     input_text = topic + " [SEP] " + argument
#
#     # Tokenize the input
#     inputs = tokenizer_tuned(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
#
#     # Move the inputs to the same device as the model
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#
#     # Predict with the model
#     with torch.no_grad():
#         outputs = model_tuned(**inputs)
#         # Get the predicted score
#         score = outputs.logits.squeeze().item()
#
#     return score
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
with open(
        "/scratch/rrs99/Stance_Distribution/community_detection_main_classes/output_survey_" + cur_topic + "_allsides_200_articles_discourse_arguments.json",
        "r") as file_j:

    loaded_data = json.load(file_j)
    topic_word = loaded_data['topic']
    eva_communities = loaded_data["eva_communities"]
    predicted_graph = json_graph.node_link_graph(loaded_data["predicted_graph"])
    perspective_outputs, perspective_labels = loaded_data["community_args"]
    all_data = []
    final_result = {}
    for i, subgraph_nodes in enumerate(eva_communities):
        subgraph = predicted_graph.subgraph([str(node) for node in subgraph_nodes])
        # for node in subgraph.nodes():
        #     print(node, subgraph.nodes[node]['topic'])
        #     break
        # if len(subgraph_nodes) <= 5:
        #     continue
        # extract_subgraph_with_properties(subgraph)
        # print("community: ", i + 1)
        # print(subgraph.edges(data=True))
        for node in subgraph.nodes():
            all_data.append((i, node, subgraph.nodes[node]['text'], topic))

    print(f"len of all data: {len(all_data)}")
    test_df = pd.DataFrame(all_data, columns=['community_index', 'node_index', 'argument', 'topic'])
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    test_dataloader = DataLoader(test_dataset, batch_size=8)

    test_df['answer'] = evaluate(model_tuned, test_dataloader)
    # json_file_path = '/scratch/rrs99/Stance_Distribution/community_detection_main_classes/' + cur_topic + '_arg_quality_roberta.json'
    # csv_file_path = '/scratch/rrs99/Stance_Distribution/community_detection_main_classes/' + cur_topic + '_arg_quality_roberta.csv'
    json_file_path = '/scratch/rrs99/Stance_Distribution/community_detection_main_classes/' + cur_topic + '_survey__arg_quality_roberta.json'
    csv_file_path = '/scratch/rrs99/Stance_Distribution/community_detection_main_classes/' + cur_topic + '_survey__arg_quality_roberta.csv'

    test_df.to_csv(csv_file_path, index=False)
    test_df.to_json(json_file_path, orient='records')
