import pytest
import json
from scripts.preprocessing.vector_storage import VectorStorage

@pytest.fixture
def setup_vector_storage():
    """
    Set up the vector store and populate it with sample documents.
    """
    vector_storage = VectorStorage(
        index_path="./data/sample_vector_store.faiss",
        mapping_path="./data/sample_celex_mapping.json"
    )

    with open("./data/sample_docs.json", "r") as f:
        documents = json.load(f)
    
    vector_storage.create_index(documents)
    return vector_storage

def test_retrieve_exact_match(setup_vector_storage):
    """
    Test 1:1 retrieval for each document in the vector store.
    """
    vector_storage = setup_vector_storage

    with open("./data/sample_docs.json", "r") as f:
        documents = json.load(f)

    for doc in documents:
        query_text = doc["text"]
        expected_celex_id = doc["celex_id"]

        # Retrieve top 1 result
        results = vector_storage.retrieve_documents(query=query_text, k=1)
        retrieved_celex_id, _ = results[0]

        assert retrieved_celex_id == expected_celex_id, f"Failed for CELEX ID: {expected_celex_id}"
