import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader

from config import LLM_REPO_ID, DATA_DIRECTORY
from chunking import setup_parent_child_retriever
from retrieval import build_advanced_retriever

def start_bot():
    print("Initializing Professional Recipe RAG...")
    
    # 1. Setup Base Retrievers
    base_retriever = setup_parent_child_retriever()
    
    # Note: BM25 needs the original docs in memory to search keywords
    loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
    docs = loader.load()
    
    # 2. Build Advanced Search Pipeline
    advanced_retriever = build_advanced_retriever(docs, base_retriever)
    
    # 3. Setup LLM
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID, 
        task="conversational",
        temperature=0.1, 
        max_new_tokens=512,
        huggingfacehub_api_token=token
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    prompt = PromptTemplate.from_template("""
    You are an expert chef assistant. Use the provided recipe context to answer the user.
    If the ingredients aren't in the context, say you don't know.

    Context: {context}
    Question: {question}
    Answer:""")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=advanced_retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    print("\n--- Chef Bot Ready! Type 'exit' to quit. ---")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
            
        result = qa_chain.invoke({"query": query})
        print(f"\nChef Bot: {result['result']}")

if __name__ == "__main__":
    start_bot()