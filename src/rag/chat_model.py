import os
import yaml
from typing import cast
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import Document
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import chainlit as cl
from langchain import hub
from src.rag.retriever import load_vectorstore, retrieval_qa_chain

# Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'
DEFAULT_CONFIG_PATH = "chainlit.yaml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load configuration from YAML file, or use defaults if not found."""
    default_config = {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "quantization": "4bit",
        "repetition_penalty": 1.2,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95
    }
    
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            user_config = yaml.safe_load(file)
        print(f"Loaded configuration from {config_path}")
        return {**default_config, **user_config}  # Merge user config with defaults
    else:
        print("Using default configuration.")
        return default_config

def create_huggingface_endpoint(args) -> HuggingFacePipeline:
    """
    Load a Hugging Face model with configurable settings, ensuring CPU fallback.
    """
    try:
        use_cpu = not torch.cuda.is_available()  # Detect if GPU is available

        quant_config = None
        if args["quantization"] == "4bit" and not use_cpu:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif args["quantization"] == "8bit" and not use_cpu:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Force model to CPU if no GPU is available
        device_map = "auto" if not use_cpu else {"": "cpu"}

        model = AutoModelForCausalLM.from_pretrained(
            args["model_name"],
            quantization_config=quant_config if not use_cpu else None,
            device_map=device_map
        )

        tokenizer = AutoTokenizer.from_pretrained(args["model_name"])

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args["max_new_tokens"],
            temperature=args["temperature"],
            repetition_penalty=args["repetition_penalty"],
            do_sample=args["do_sample"],
            top_k=args["top_k"],
            top_p=args["top_p"],
        )

        return HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        print(f"Error loading model: {e}")
        raise ValueError(f"Failed to load model {args['model_name']}: {e}")

def set_custom_prompt():
    """Create a prompt template for QA retrieval."""
    return hub.pull("langchain-ai/retrieval-qa-chat")

@cl.on_chat_start
async def start():
    msg = cl.Message(content="Initializing RAG Bot... Please wait.")
    await msg.send()

    try:
        msg.content = "Loading configuration..."
        await msg.update()
        args = load_config()
        print(f"Using configuration: {args}")

        msg.content = "Loading the language model... This may take a few minutes."
        await msg.update()
        llm = await cl.make_async(create_huggingface_endpoint)(args)

        msg.content = "Loading the vector database..."
        await msg.update()
        db = await cl.make_async(load_vectorstore)(args)

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

    if runnable is None:
        await message.reply("Error: RAG Bot is not initialized properly. Please restart.")
        return

    msg = cl.Message(content="")
    text_elements = []

    async for chunk in runnable.astream(
        {"input": message.content, "vectorstore": runnable.vectorstore},
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