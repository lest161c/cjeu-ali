import torch
import re
import torch.nn.functional as F

def extract_celex_ids(text):
    """
    Extract CELEX IDs from text using regex.
    """
    #TODO: needs refinement
    return re.findall(r'319\d{7}', text)

def citation_loss(predicted_text, retrieved_celex_ids):
    """
    Custom loss for penalizing incorrect or missing CELEX citations.
    """
    predicted_ids = extract_celex_ids(predicted_text)

    # Calculate penalties
    missing_ids = set(retrieved_celex_ids) - set(predicted_ids)
    extra_ids = set(predicted_ids) - set(retrieved_celex_ids)

    return len(missing_ids) + len(extra_ids)

# Example usage
if __name__ == "__main__":
    predicted = "The decision references 31979D0509 and 31979D0510."
    retrieved_ids = ["31979D0509", "31979D0511"]
    loss = citation_loss(predicted, retrieved_ids)
    print("Citation Loss:", loss)
