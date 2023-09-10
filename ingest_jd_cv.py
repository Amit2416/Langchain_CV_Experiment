from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
import os 
#from constants import CHROMA_SETTINGS

# Define the directories for CVs and Job Descriptions
cv_directory = "docs/CV"
job_description_directory = "docs/job_description"

# Create separate persist directories for CVs and Job Descriptions
cv_persist_directory = "cv_db"
job_description_persist_directory = "job_description_db"

def process_documents(input_directory, persist_directory):
    for file in os.listdir(input_directory):
        if file.endswith(".pdf"):
            print(f"Processing {os.path.join(input_directory, file)}")
            loader = PDFMinerLoader(os.path.join(input_directory, file))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
            db.persist()
            db = None

def main():
    # Process CVs
    process_documents(cv_directory, cv_persist_directory)

    # Process Job Descriptions
    process_documents(job_description_directory, job_description_persist_directory)

if __name__ == "__main__":
    main()