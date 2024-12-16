from scripts.preprocessing.vector_storage import VectorStorage

class RetrievalAugmentor:
    def __init__(self, vector_storage_path='./data/legal_doc_index.faiss'):
        self.vector_storage = VectorStorage(index_path=vector_storage_path)

    def augment_input(self, input_text, k=3):
        """
        Retrieve relevant context from vector storage and append it to the input text.
        """
        retrieved_docs = self.vector_storage.retrieve_documents(input_text, k=k)
        context = "\n\n".join([doc[1] for doc in retrieved_docs])
        return f"{input_text}\n\n{context}"

# Example usage
if __name__ == "__main__":
    retrieval_augmentor = RetrievalAugmentor()
    input_text = "Community financial aid for swine fever eradication"
    augmented_text = retrieval_augmentor.augment_input(input_text)
    print("Augmented Input:", augmented_text)
