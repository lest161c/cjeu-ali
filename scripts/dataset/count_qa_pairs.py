import json

dataset_path = "data/q_a_dataset.json"
with open(dataset_path, "r") as f:
    q_a_pairs = json.load(f)
    number_of_q_a_pairs = len(q_a_pairs)
    print(f"{number_of_q_a_pairs} Q&A pairs loaded from {dataset_path}")