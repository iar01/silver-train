from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from database import get_vector_db

def setup_parent_child_retriever():
    """Creates a retriever that searches small chunks but returns full recipes."""
    vector_db = get_vector_db()
    store = InMemoryStore()
    
    # Parent chunks (The whole recipe context)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    
    # Child chunks (Strict ingredient/step matching)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    
    retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever