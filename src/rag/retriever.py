import torch
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict, Any

@chain
def retriever(inputs: Dict[str, Any]) -> List[Document]:
    """
    Retrieve relevant documents for a given query.
    
    Args:
        inputs (Dict[str, Any]): A dictionary containing the query and other parameters.
    
    Returns:
        List[Document]: List of relevant documents.
    """
    vectorstore = inputs['vectorstore']
    query = inputs['input']
    k = inputs.get('k', None)  # Get 'k' from inputs, default to None if not provided

    docs = []
    if k is not None:
        docs, scores = zip(*vectorstore.similarity_search_with_relevance_scores(query, k=k))
    else: 
        docs, scores = zip(*vectorstore.similarity_search_with_score(query))

    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

class SimilarityScoreVectorStoreRetriever(BaseRetriever):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def _get_relevant_documents(self, query: str, k: int = None) -> List[Document]:
        if k is not None:
            docs, scores = zip(*self.vectorstore.similarity_search_with_relevance_scores(query, k=k))
        else:
            docs, scores = zip(*self.vectorstore.similarity_search_with_score(query))

        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = create_stuff_documents_chain(llm, prompt)

    retriever = SimilarityScoreVectorStoreRetriever(db)

    return create_retrieval_chain(retriever, qa_chain)

def retrieve_related_documents(db, question, args):
    if args["use_similarity_threshold"]:
        input_args = {"vectorstore": db, "input": question}
        docs = retriever.invoke(input_args)
        return [doc for doc in docs if doc.metadata.get('score', 0) >= args["similarity_threshold"]]
    else:
        input_args = {"vectorstore": db, "input": question, "k": args["k"]}
        docs = retriever.invoke(input_args)

def load_vectorstore(args):
    """Load FAISS vectorstore using Hugging Face Embeddings, ensuring CPU fallback."""
    try:
        use_cpu = not torch.cuda.is_available()
        device = "cpu" if use_cpu else "cuda"

        print(f"Using device for embeddings: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=args["embedding_model_name"],
            model_kwargs={
                'device': device,
                'trust_remote_code': True,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
            }
        )

        db = FAISS.load_local(
            args["db_faiss_path"], 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        return db
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise ValueError(f"Failed to load vector store: {e}")
