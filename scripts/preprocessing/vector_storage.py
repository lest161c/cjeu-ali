import faiss
import json
import os
from sentence_transformers import SentenceTransformer

class VectorStorage:
    def __init__(self, index_path, mapping_path, embedding_model="all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(384)  # 384-dim embeddings for MiniLM
        self.celex_mapping = []

    def create_index(self, documents):
        """
        Add documents to the vector store and save the index.
        """
        texts = [doc["text"] for doc in documents]
        celex_ids = [doc["celex_id"] for doc in documents]

        embeddings = self.model.encode(texts, convert_to_tensor=False)
        self.index.add(embeddings)
        self.celex_mapping.extend(celex_ids)

        # Save the index and mapping
        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "w") as f:
            json.dump(self.celex_mapping, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, "r") as f:
                self.celex_mapping = json.load(f)
        else:
            raise FileNotFoundError("Index or mapping file not found.")

    def retrieve_documents(self, query, k=1):
        """
        Retrieve the top-k most similar documents.
        """
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.celex_mapping):
                results.append((self.celex_mapping[idx], dist))
        
        return results
