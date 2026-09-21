import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from chunking import setup_parent_child_retriever
from config import DATA_DIRECTORY

def ingest_data():
    print("1. Loading Recipe PDFs...")
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
        print(f"Please put PDFs in the {DATA_DIRECTORY} folder and run again.")
        return

    loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
    documents = loader.load()
    
    if not documents:
        print("No documents found!")
        return

    print("2. Chunking and saving to Vector DB (Parent-Child Strategy)...")
    retriever = setup_parent_child_retriever()
    
    # This automatically splits and saves to ChromaDB and our in-memory store
    retriever.add_documents(documents)
    
    print("SUCCESS: Data ingested! You can now run main.py")

if __name__ == "__main__":
    ingest_data()