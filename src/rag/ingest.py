
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader
from tqdm import tqdm
import faiss

DATA_PATH = 'data/eurlex_full.json'
DB_FAISS_PATH = 'vectorstore/db_faiss'


def add_celex_id_to_metadata(r: dict, m: dict) -> dict:
    m["celex_id"] = r["celex_id"]
    m["title"] = r["title"]
    m["eurovoc_concepts"] = r["eurovoc_concepts"]
    return m

# Create vector database
def create_vector_db():
    # Load documents
    loader = JSONLoader(DATA_PATH,
                        jq_schema=".documents[]",
                        text_content=False,
                        content_key="text",
                        metadata_func=add_celex_id_to_metadata
                       )

    documents = loader.load()
    
    # Check if documents are loaded
    if not documents:
        raise ValueError("No documents found in the provided JSON file.")

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2', 
        model_kwargs={'device': 'cpu'}
    )
    
    # Create the FAISS vector store
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    celex_ids = [doc.metadata['celex_id'] for doc in documents]
    # Add documents to the FAISS database
    db.add_documents(documents=documents, ids=celex_ids)
    
    # Save FAISS database locally
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS database saved at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
