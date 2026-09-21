from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import DB_DIRECTORY, EMBEDDING_MODEL

def get_vector_db():
    """Initializes and returns the ChromaDB connection."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    db = Chroma(
        collection_name="recipes",
        persist_directory=DB_DIRECTORY, 
        embedding_function=embeddings
    )
    return db