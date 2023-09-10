from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
#from constants import CHROMA_SETTINGS

#model and tokenizer loading
checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

def llm_pipeline():
        """
    Create a Language Model (LLM) pipeline for text generation.

    Returns:
        HuggingFacePipeline: A HuggingFace LLM pipeline for text generation.
    """
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 512,
        do_sample=True,
        temperature = 0.01,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_llm():
        """
    Create a Question-Answering (QA) Language Model (LLM) pipeline.

    Returns:
        RetrievalQA: A RetrievalQA pipeline for question-answering.
    """
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="cv_db", embedding_function=embeddings)#, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def load_responses(file_path):
        """
    Load responses from a text file and extract them.

    Args:
        file_path (str): The path to the text file containing responses.

    Returns:
        list: A list of response strings.
    """
    responses = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("Response:"):
                response = line.strip().replace("Response:", "Does candidate has:").strip()
                responses.append(response)
    return responses

max_sequence_length = 512

def truncate_text(text):
        """
    Truncate text to a maximum sequence length.

    Args:
        text (str): The input text.

    Returns:
        str: The truncated text.
    """
    if len(text) > max_sequence_length:
        text = text[:max_sequence_length]
    return text

def process_answer(responses):
        """
    Process responses using the QA Language Model (LLM) pipeline.

    Args:
        responses (list): A list of response strings.

    Returns:
        list: A list of generated answers.
    """
    qa = qa_llm()
    results = []

    for response in responses:
        question = truncate_text(response)
        generated_text = qa(question)
        answer = generated_text['result']
        results.append(answer)
    
    return results




def main():
    response_file_path = "response_jd.txt"
    responses = load_responses(response_file_path)
    results = process_answer(responses)

    # Create a new text file to save the answers along with the original information
    output_file_path = "generated_answers.txt"
    with open(output_file_path, "w") as output_file:
        for i, response in enumerate(responses):
            original_info = f"Response {i + 1}:\n{response}\n"
            generated_answer = f"Generated Answer {i + 1}:\n{results[i]}\n"
            output_file.write(original_info)
            output_file.write(generated_answer)

if __name__ == "__main__":
    main()

