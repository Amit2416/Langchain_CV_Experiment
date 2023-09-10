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
        max_length = 256,
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
    db = Chroma(persist_directory="job_description_db", embedding_function=embeddings)#, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
        """
    Generate a response to a given instruction using the QA Language Model (LLM) pipeline.

    Args:
        instruction (str): The instruction or question for which a response is generated.

    Returns:
        str: The generated response.
    """
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


if __name__ == '__main__':
    with open("prompt_jd.txt", "r") as file:
        instructions = file.readlines()

    with open("response_jd.txt", "w") as output_file:
        for i, instruction in enumerate(instructions):
            instruction = instruction.strip()
            print(f"Processing Instruction {i + 1}: {instruction}")
            answer = process_answer(instruction)
            print(f"Response: {answer}")
            output_file.write(f"Instruction {i + 1}: {instruction}\n")
            output_file.write(f"Response: {answer}\n")

