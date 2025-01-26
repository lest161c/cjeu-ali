import os
import json
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
SCADS_API_KEY = os.getenv("SCADS_API_KEY")

def _get_q_a_pairs_from_gippity(doc, client, model_name):
    # Ask GPT nicely for QA pairs for current document :)
    sys_msg = '''
    You are an assistant that figures out when the verdict in a provided document was decided (this information is in the document text). 
    Your answer should be in the format: DD-MM-YYYY
    '''

    response = client.chat.completions.create(
    model   =  model_name,
    messages=[
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": doc}
    ]
    )
    response_text = response.choices[0].message.content
    
    formatted_response = [
    {
        "question": "exQ",
        "answer": "exA"
    },
    {
        "question": "exQ",
        "answer": "exA"
    }
    ]

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

    # Format of json file is a list containing strings
    q_a_pairs = []
    for doc in data:
        response = _get_q_a_pairs_from_gippity(doc, client, model_name)
        for qa in response:
            q_a_pairs.append(qa)


get_q_a_dataset("data/q_a_dataset.json", "filtered_doc/extracted_documents.json")