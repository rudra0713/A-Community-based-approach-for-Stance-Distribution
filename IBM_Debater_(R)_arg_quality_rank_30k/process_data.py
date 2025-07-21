import pandas as pd


csv_file_path = "arg_quality_rank_30k.csv"
df = pd.read_csv(csv_file_path)
df['WA'] = df['WA'].apply(lambda x: 'important' if x >= 0.7 else 'not important')

# Create three new DataFrames based on the "set" column values
X_train = df[df['set'] == 'train']
X_eval = df[df['set'] == 'dev']
X_test = df[df['set'] == 'test']
X_train = X_train.head(1000)
X_eval = X_eval.head(1000)


def generate_prompt(data_point):
    return f"""
            Classify the argument as to whether it is "important" or "not important" with respect to the topic.
            argument: {data_point["argument"]}
            topic: {data_point["topic"]}            
            label: {data_point["WA"]}""".strip()


def generate_test_prompt(data_point):
    return f"""
            Classify the argument as to whether it is "important" or "not important" with respect to the topic.
            argument: {data_point["argument"]}
            topic: {data_point["topic"]}            
            label: """.strip()


# Generate prompts for training and evaluation data
X_train.loc[:, 'text'] = X_train.apply(generate_prompt, axis=1)
X_eval.loc[:, 'text'] = X_eval.apply(generate_prompt, axis=1)

value_counts_train = X_train['WA'].value_counts()
value_counts_eval = X_eval['WA'].value_counts()

print(value_counts_train)
print(value_counts_eval)

