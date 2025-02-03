import os
import argparse
import torch
import chainlit as cl
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain import hub

DB_FAISS_PATH = 'vectorstore/db_faiss'

def parse_args():
    parser = argparse.ArgumentParser(description="Configurable RAG Bot")

    # Model Configuration
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                        help="Hugging Face model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens for text generation")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], default="4bit", help="Quantization level")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repetition")
    parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-P sampling")

    return parser.parse_args()

def create_huggingface_endpoint(args) -> HuggingFacePipeline:
    """Load a Hugging Face model with configurable settings."""
    try:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        quant_config = None

        if args.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif args.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise ValueError(f"Failed to load model {args.model_name}: {e}")

def set_custom_prompt():
    """Create a prompt template for QA retrieval."""
    return hub.pull("langchain-ai/retrieval-qa-chat")

def load_vectorstore():
    """Load FAISS vectorstore using Hugging Face Embeddings."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for embeddings: {device}")
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={
                'device': device,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
            }
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise ValueError(f"Failed to load vector store: {e}")

def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 1})
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

@cl.on_chat_start
async def start():
    msg = cl.Message(content="Initializing RAG Bot... Please wait.")
    await msg.send()

    try:
        msg.content = "Loading configuration..."
        await msg.update()

        args = parse_args()

        msg.content = "Loading the language model... This may take a few minutes."
        await msg.update()

        llm = await cl.make_async(create_huggingface_endpoint)(args)
        msg.content = "Loading the vector database..."
        await msg.update()
        db = await cl.make_async(load_vectorstore)()

        msg.content = "Configuring the retrieval chain..."
        await msg.update()
        qa_prompt = set_custom_prompt()
        qa_chain = retrieval_qa_chain(llm, qa_prompt, db)

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
        if "input" in chunk:
            continue
        elif "context" in chunk:
            source_documents = chunk.get("context", [])
            for source_idx, source_doc in enumerate(source_documents):
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
            await msg.stream_token(str(chunk["answer"]))

    if text_elements:
        source_names = [text_el.name for text_el in text_elements]
        msg.content += f"\nSources: {', '.join(source_names)}"
        msg.elements = text_elements
    
    await msg.send()
