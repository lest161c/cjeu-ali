import json
from vector_storage import VectorStorage

def populate_vector_store():
    """
    Load the sample dataset and populate the vector store.
    """
    vector_storage = VectorStorage(
        index_path="./data/sample_vector_store.faiss",
        mapping_path="./data/sample_celex_mapping.json"
    )

    with open("./data/sample_docs.json", "r") as f:
        documents = json.load(f)
    
    vector_storage.create_index(documents)
    print("Vector store populated successfully.")

if __name__ == "__main__":
    populate_vector_store()
