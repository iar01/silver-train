from torch import embedding
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def create_database():
    print("1. Loading PDFs....")
    documents = []
    data_dir = "data"
    
    pdf_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ] if os.path.exists(data_dir) else []

    if not pdf_files:
        print("No PDF files found in data folder.")
        return

    for file in pdf_files:
        filename = os.path.basename(file)
        print(f"  -> Loading {filename}...")
        try:
            loader = PyMuPDFLoader(file)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            # print(f"     [OK] Loaded {len(loaded_docs)} pages.")
        except Exception as e:
            print(f"     [ERROR] Failed to load {filename}: {e}")
        
    if not documents:
        print("No documents could be loaded.")
        return

    # print(f"\nCompleted loading: {len(documents)} total pages from {len(pdf_files)} PDFs.")

    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", ""],
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    # print(f"\nCreated {len(chunks)} chunks.")

    print("\n2. Creating Embeddings using HuggingFaceEmbeddings...")

    embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vector_db= Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("\nSUCCESS: Vector database created and saved locally in './chroma_db'!")

if __name__ == "__main__":
    create_database()