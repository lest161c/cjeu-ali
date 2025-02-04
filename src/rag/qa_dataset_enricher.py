import os
import json
import argparse
from tqdm import tqdm
from src.rag.retriever import load_vectorstore, retrieve_related_documents

def parse_args():
    parser = argparse.ArgumentParser(description="Enrich dataset using RAG.")
    parser.add_argument("--db_faiss_path", type=str, default='vectorstore/db_faiss', help="Path to FAISS vectorstore.")
    parser.add_argument("--input_dataset", type=str, default='dataset.jsonl', help="Path to input dataset (JSONL file).")
    parser.add_argument("--output_dataset", type=str, default='enriched_dataset.jsonl', help="Path to output dataset (JSONL file).")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of documents to process per batch.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of related documents to retrieve.")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="Similarity score threshold.")
    parser.add_argument("--use_similarity_threshold", action="store_true", help="Use similarity threshold instead of top K.")
    parser.add_argument("--progress_file", type=str, default='progress.json', help="Path to progress tracking file.")
    return parser.parse_args()

def load_dataset(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f][0]

def save_progress(progress_file, last_processed_id):
    with open(progress_file, 'w') as f:
        json.dump({"last_processed_id": last_processed_id}, f)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f).get("last_processed_id", -1)
    return -1

def process_dataset(args):
    db = load_vectorstore({"db_faiss_path": args.db_faiss_path, "embedding_model_name": "BAAI/bge-large-en-v1.5"})
    dataset = load_dataset(args.input_dataset)
    last_processed_id = load_progress(args.progress_file)
    
    with open(args.output_dataset, 'a', encoding='utf-8') as f_out:
        for i in tqdm(range(last_processed_id + 1, len(dataset)), desc="Processing documents"):
            item = dataset[i]
            question = item["question"]
            answer = item.get("answer", "")
            
            search_kwargs = {"top_k": args.top_k,
                             "use_similarity_threshold": args.use_similarity_threshold,
                             "similarity_threshold": args.similarity_threshold}
            related_docs = retrieve_related_documents(db, question, search_kwargs)
            related_texts = "\n".join([f"{idx+1}: {doc.page_content}" for idx, doc in enumerate(related_docs)])
            
            enriched_question = f"Question: {question}\n\nRelated Documents:\n{related_texts}"
            
            enriched_entry = json.dumps({"question": enriched_question, "answer": answer})
            f_out.write(enriched_entry + "\n")
            
            if (i + 1) % args.batch_size == 0:
                save_progress(args.progress_file, i)
    
    save_progress(args.progress_file, len(dataset) - 1)

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)