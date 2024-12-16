from datasets import load_dataset
import json
import os

def load_eurlex57k_sample(output_path, num_samples=5):
    """
    Load a small subset of the EURLEX57K dataset and save it as JSON.
    """
    dataset = load_dataset("NLP-AUEB/eurlex", split="train[:5]")
    sample_data = []

    for doc in dataset:
        sample_data.append({
            "celex_id": doc["celex_id"],
            "text": doc["text"]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample dataset saved to {output_path}")

if __name__ == "__main__":
    load_eurlex57k_sample("./data/sample_docs.json")
