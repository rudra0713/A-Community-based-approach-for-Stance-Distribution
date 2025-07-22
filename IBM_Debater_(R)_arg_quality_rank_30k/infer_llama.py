from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from networkx.readwrite import json_graph
import json, torch, argparse
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from peft import LoraConfig, PeftConfig, PeftModel


def parse_arguments():
    parser = argparse.ArgumentParser('Argument Quality')
    parser.add_argument('--dir_index', type=str, default="1")

    return parser.parse_args()


args = parse_arguments()
print(f"args.dir_index: ", args.dir_index)


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

login(token=os.getenv("HF_TOKEN"))
access_token = os.getenv("HF_TOKEN")


def generate_test_prompt(data_point):
    return f"""
            Classify the argument as to whether it is "relevant" or "irrelevant" with respect to the topic.
            argument: {data_point["argument"]}
            topic: {data_point["topic"]}            
            label: """.strip()


def predict(test, model, tokenizer):
    y_pred = []
    categories = ["relevant", "irrelevant"]

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=2,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        # print(f"prompt: {prompt}")
        # print(f"answer: {answer}")
        # Determine the predicted category
        for category in categories:
            if category.lower() == answer.lower():
                y_pred.append(category)
                break
        else:
            print(f"unknown answer: {answer}")
            y_pred.append("none")
    print(f"final y pred: ")
    print(y_pred)
    return y_pred


topic_list = ['abortion', 'elections', 'gun_control_and_gun_rights', 'immigration', 'taxes']

cur_topic = topic_list[int(args.dir_index) - 1]
topic = None
if cur_topic == 'abortion':
    topic = 'abortion'
elif cur_topic == 'elections':
    topic = '2024 presidential Election'
elif cur_topic == 'gun_control_and_gun_rights':
    topic = 'Gun Control and Gun Rights'
elif cur_topic == 'immigration':
    topic = 'Immigration'
elif cur_topic == 'taxes':
    topic = 'Taxes'

print(f"focusing on topic: {topic}")
base_model = 'meta-llama/Meta-Llama-3-8B'
fine_tuned_model = "/scratch/rrs99/Stance_Distribution/IBM_Debater_(R)_arg_quality_rank_30k/llama-3.1-fine-tuned-model"

tokenizer = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)

# Merge adapter with base model
model = PeftModel.from_pretrained(base_model_reload, fine_tuned_model)
model = model.merge_and_unload()


with open("/scratch/rrs99/Stance_Distribution/community_detection_main_classes/output_" + cur_topic + "_allsides_200_articles_discourse_arguments.json", "r") as file:
    loaded_data = json.load(file)
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
        if len(subgraph_nodes) <= 5:
            continue
        # extract_subgraph_with_properties(subgraph)
        # print("community: ", i + 1)
        # print(subgraph.edges(data=True))
        for node in subgraph.nodes():
            all_data.append((i, node, subgraph.nodes[node]['text'], topic))

    print(f"len of all data: {len(all_data)}")
    X_test = pd.DataFrame(all_data, columns=['community_index', 'node_index', 'argument', 'topic'])
    X_test['text'] = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1))
    y_pred = predict(X_test, model, tokenizer)
    print(f"len of y_pred: {len(y_pred)}")
    X_test['answer'] = y_pred
    json_file_path = '/scratch/rrs99/Stance_Distribution/community_detection_main_classes/' + cur_topic + '_arg_quality.json'
    csv_file_path = '/scratch/rrs99/Stance_Distribution/community_detection_main_classes/' + cur_topic + '_arg_quality.csv'
    X_test.to_csv(csv_file_path, index=False)
    X_test.to_json(json_file_path, orient='records')
