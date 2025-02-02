import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re


load_dotenv()
SCADS_API_KEY = os.getenv("SCADS_API_KEY")

def _get_q_a_pairs_from_dalai_llama(doc, client, model_name):
    # Ask GPT nicely for QA pairs for current document :)
    sys_msg = '''
    You will be provided with a ruling from the European Court of Justice. Your task is to transform the document into a structured set of question-answer pairs. Follow these instructions carefully:

    Extract Legal Interpretations:
        Identify key legal questions that the court addresses.
        Ensure questions focus on legal interpretations rather than factual case details.

    Format Output as Question-Answer Pairs:
        Present questions and answers in the format:

        Q1. [Example Question One]  
        A1. [Example Answer One]  

        The number of Q&A pairs may vary based on the complexity of the case.

    Maintain Precision & Fidelity:
        The answers should be direct excerpts or clear summaries of the court's ruling on each legal question.
        Do not introduce interpretations beyond the ruling itself.

    Improve Legal Clarity:
        If necessary, refine the phrasing of the questions to make them clearer while preserving the original legal issue.
        Ensure that answers align exactly with the court's conclusions.
        Format the questions and answers consistently in a way that is agnostic to the specific case.
        Exclude terms such as defendant, plaintiff, etc. from the response unless necessary context is given in the question.
        The question answer pairs should be atomic, the answer should be reproduceable from that question in a vacuum.
        Be clear about specific things. For example do not say "the court" say "the European Court of Justice", or "the treaty" say "the Treaty of Lisbon".
        Where possible refer to name of things instead of numbers. For example do not say "Article 30" say "Article 30 of the Treaty on the Functioning of the European Union".

    Example Output:

    Input: European Court of Justice ruling on data privacy under GDPR
    Output:
    Q1. Does GDPR allow national laws to impose additional data processing restrictions beyond what is stated in EU law?  
    A1. The court rules that while a Member State may introduce supplementary safeguards, they cannot contradict or restrict the fundamental principles established by GDPR.  

    Q2. Under what conditions can a company be fined for inadequate data security measures?  
    A2. The ruling states that fines can be imposed when a company fails to implement "appropriate technical and organizational measures" as required by Article 32 of GDPR, leading to unauthorized data access.  

    By following these steps, generate well-structured Q&A pairs based on the provided court ruling.
    '''

    response = client.chat.completions.create(
    model   =  model_name,
    messages=[
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": doc}
    ]
    )
    response_text = response.choices[0].message.content

    questions = re.findall(r"Q\d+\.\s(.+)", response_text)
    answers = re.findall(r"A\d+\.\s(.+)", response_text)
    
    formatted_response = []
    for q, a in zip(questions, answers):
        formatted_response.append({"question": q, "answer": a})

    return formatted_response

def get_q_a_dataset(output_path, input_path):
    """
    Load json containing all CJEU rulings from input_path and extract questions and answers using scads llm api.
    Save all question answer pairs in a json file at output_path.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    client = OpenAI(base_url="https://llm.scads.ai/v1",api_key=SCADS_API_KEY)
    for model in client.models.list().data:
        model_name = model.id
        if "llama" in model_name:
            break

    # Format of input file is a list containing strings
    docs_already_processed = 400
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            q_a_pairs = json.load(f)
    else:
        q_a_pairs = []

    for i, doc in enumerate(tqdm(data)):
        if i < docs_already_processed:
            continue
        if i % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(q_a_pairs, f)
            print(f"Saved {i} documents")
        response = _get_q_a_pairs_from_dalai_llama(doc, client, model_name)
        for qa in response:
            q_a_pairs.append(qa)

    # Format of output file is a list containing dictionaries with keys "question" and "answer"
    with open(output_path, "w") as f:
        json.dump(q_a_pairs, f)


get_q_a_dataset("data/q_a_dataset.json", "filtered_doc/extracted_documents.json")