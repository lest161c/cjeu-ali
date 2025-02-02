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
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

import chainlit as cl
from langchain import hub

# Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_huggingface_endpoint(
    model_name: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2
) -> HuggingFacePipeline:
    """
    Create a local Hugging Face pipeline endpoint for LangChain.
    
    Args:
        model_name (str): The name of the Hugging Face model to load.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.2.
    
    Returns:
        HuggingFacePipeline: Configured model pipeline
    """
    try:
        # Verify CUDA availability
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        
        # Load model and tokenizer with accelerate-friendly settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            low_cpu_mem_usage=True,     # Reduce CPU memory usage
            # No explicit device_map needed with accelerate
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        
        # Convert to HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return llm
    
    except Exception as e:
        print(f"Error creating Hugging Face endpoint: {e}")
        raise ValueError(f"Failed to load model {model_name}: {e}")

def set_custom_prompt():
    """
    Create a prompt template for QA retrieval.
    """
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    return retrieval_qa_chat_prompt

# Load Embeddings and Vectorstore

def load_vectorstore():
    """
    Load FAISS vectorstore using Hugging Face Embeddings.
    Automatically detects and uses available device.
    
    Returns:
        FAISS: Loaded vector store
    """
    try:
        # Determine the best available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Print device information for debugging
        print(f"Using device for embeddings: {device}")
        
        # Create embeddings with device-aware configuration
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={
                'device': device,
                # Additional optimization parameters
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            },
            encode_kwargs={
                'normalize_embeddings': True,  # Recommended for BGE models
                'batch_size': 32,  # Adjust based on your GPU memory
            }
        )
        
        # Load the vector store
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        return db
    
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise ValueError(f"Failed to load vector store: {e}")

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 1})
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    return retrieval_chain

# QA Bot Function
def qa_bot():
    llm = create_huggingface_endpoint("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    db = load_vectorstore()
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
    msg = cl.Message(content="Initializing RAG Bot... Please wait.")
    await msg.send()

    try:
        # Notify the user that model loading has started
        msg.content = "Loading the language model... This may take a few minutes."
        await msg.update()

        # Initialize the model asynchronously
        llm = await cl.make_async(create_huggingface_endpoint)(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        )

        msg.content = "Loading the vector database..."
        await msg.update()
        db = await cl.make_async(load_vectorstore)()

        msg.content = "Configuring the retrieval chain..."
        await msg.update()
        qa_prompt = set_custom_prompt()
        qa_chain = retrieval_qa_chain(llm, qa_prompt, db)

        # Update the message when everything is ready
        msg.content = "Hi, Welcome to RAG Bot! What is your query?"
        await msg.update()

        cl.user_session.set("runnable", qa_chain)

    except Exception as e:
        msg.content = f"Error initializing bot: {e}"
        await msg.update()


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