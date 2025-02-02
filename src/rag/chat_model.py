import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import cast
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import Document
import torch
from transformers import AutoModelForCausalLM
from typing import Optional, Union, Dict, Any
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

import chainlit as cl
from langchain import hub

# Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_huggingface_endpoint(
    model_name: str,
) -> HuggingFaceEndpoint:
    """
    Create a configurable Hugging Face endpoint.
    
    Args:
        model_name (str): The name of the Hugging Face model to load.
    Returns:
        HuggingFaceEndpoint: Configured model endpoint
    """
    
    # Create endpoint
    try:
        endpoint = HuggingFaceEndpoint(
            max_new_tokens=1024,
            repo_id=model_name,
            temperature=0.7,
            # Add timeout and streaming parameters
            timeout=120,  # Increase timeout to 120 seconds
            repetition_penalty=1.2,
            streaming=True,
        )
    
    except Exception as e:
        raise ValueError(f"Error creating Hugging Face endpoint: {e}")
    
    return endpoint

def set_custom_prompt():
    """
    Create a prompt template for QA retrieval.
    """
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    return retrieval_qa_chat_prompt

# Load Embeddings and Vectorstore
def load_vectorstore(device: str = 'cpu'):
    """Load FAISS vectorstore using OpenAI Embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-large-en-v1.5', 
        model_kwargs={'device': device}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 1})
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    return retrieval_chain

# QA Bot Function
def qa_bot():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    llm = create_huggingface_endpoint("meta-llama/Llama-3.2-1B-Instruct")
    db = load_vectorstore(device)
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
    return qa_chain

# Serialize Document for JSON
def serialize_document(doc):
    """Convert a Document object to a JSON-serializable dictionary."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

# Chainlit Chat Handlers
@cl.on_chat_start
async def start():
    runnable = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to RAG Bot. What is your query?"
    await msg.update()

    cl.user_session.set("runnable", runnable)

@cl.on_message
async def main(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    msg = cl.Message(content="")
    text_elements = []

    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # Process the chunk based on its content type
        if "input" in chunk:
            # Ignore "input" chunks
            continue

        elif "context" in chunk:
            # Handle context (documents)
            source_documents = chunk.get("context", [])
            for source_idx, source_doc in enumerate(source_documents):
                # Assuming `source_doc` is of type `Document`
                if isinstance(source_doc, Document):
                    source_name = f"source_{source_idx}"
                    text_elements.append(
                        cl.Text(
                            content=source_doc.page_content,
                            name=source_name,
                            display="side"
                        )
                    )

        elif "answer" in chunk:
            # Stream "answer" chunks asynchronously to the UI
            await msg.stream_token(str(chunk["answer"]))

    # Send the final message with all elements
    if text_elements:
        # Append sources to the message content
        source_names = [text_el.name for text_el in text_elements]
        msg.content += f"\nSources: {', '.join(source_names)}"
        msg.elements = text_elements
    
    await msg.send()