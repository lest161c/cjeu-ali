from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader
from tqdm import tqdm
import faiss
import logging
import os
from typing import List, Dict, Any
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = 'data/eurlex_full.json'
DB_FAISS_PATH = 'vectorstore/db_faiss'
BATCH_SIZE = 32
MODEL_NAME = "BAAI/bge-large-en-v1.5"

def add_celex_id_to_metadata(record: dict, metadata: dict) -> dict:
    """
    Add relevant fields from the record to the document metadata.
    
    Args:
        record (dict): The source record containing document information
        metadata (dict): The existing metadata dictionary
    
    Returns:
        dict: Updated metadata dictionary
    """
    metadata.update({
        "celex_id": record["celex_id"],
        "title": record["title"],
        "eurovoc_concepts": record["eurovoc_concepts"]
    })
    return metadata

def load_documents() -> List[Document]:
    """
    Load documents from the JSON file.
    
    Returns:
        List[Document]: List of loaded documents
    
    Raises:
        ValueError: If no documents are found in the JSON file
    """
    logger.info(f"Loading documents from {DATA_PATH}")
    loader = JSONLoader(
        DATA_PATH,
        jq_schema=".documents[]",
        text_content=False,
        content_key="text",
        metadata_func=add_celex_id_to_metadata
    )
    documents = loader.load()
    
    if not documents:
        raise ValueError("No documents found in the provided JSON file.")
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents

def initialize_embeddings() -> HuggingFaceBgeEmbeddings:
    """
    Initialize the embedding model.
    
    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    logger.info(f"Initializing embeddings model: {MODEL_NAME}")
    return HuggingFaceBgeEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )

def create_faiss_index(embedding_size: int) -> faiss.IndexFlatL2:
    """
    Create a FAISS index.
    
    Args:
        embedding_size (int): Size of the embeddings
    
    Returns:
        faiss.IndexFlatL2: Initialized FAISS index
    """
    return faiss.IndexFlatL2(embedding_size)

def process_documents_in_batches(
    db: FAISS,
    documents: List[Document],
    celex_ids: List[str]
) -> None:
    """
    Process documents in batches and add them to the vector store.
    
    Args:
        db (FAISS): The FAISS vector store
        documents (List[Document]): List of documents to process
        celex_ids (List[str]): List of corresponding CELEX IDs
    """
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Computing embeddings"):
        batch_end = min(i + BATCH_SIZE, len(documents))
        batched_docs = documents[i:batch_end]
        batched_celex_ids = celex_ids[i:batch_end]
        db.add_documents(documents=batched_docs, ids=batched_celex_ids)

def create_vector_db() -> None:
    """
    Main function to create and save the vector database.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        # Load documents
        documents = load_documents()
        
        # Initialize embeddings
        embeddings = initialize_embeddings()
        
        # Get embedding size using a test query
        embedding_size = len(embeddings.embed_query("test query"))
        
        # Create FAISS index
        index = create_faiss_index(embedding_size)
        
        # Initialize FAISS vector store
        db = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Extract CELEX IDs
        celex_ids = [doc.metadata['celex_id'] for doc in documents]
        
        # Process documents in batches
        process_documents_in_batches(db, documents, celex_ids)
        
        # Save the database
        db.save_local(DB_FAISS_PATH)
        logger.info(f"FAISS database successfully saved at {DB_FAISS_PATH}")
        
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_db()