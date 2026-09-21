import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

def start_chat():
    print("Loading database and AI model...")

    # 1. Load the same local embedding model
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # 2. Load the existing vector DB 
    vector_db = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)

    # 3. Create a retriever that fetches top 3 similar chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 4. Hugging Face LLM (State-of-the-art Qwen-72B supported on HF free inference)
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="conversational",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=token
    )
    chat_model = ChatHuggingFace(llm=llm)

    # 5. Create the Prompt Template
    template = """
    You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # 6. Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("\n" + "="*50)
    print("--- Free RAG Chatbot is Ready! Type 'exit' to quit. ---")
    print("="*50)

    # 7. Start the chat loop
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        # Get the answer from the RAG chain
        response = qa_chain.invoke({"query": user_query})
        
        print("\nChatbot:", response['result'])
        
        # Optional: Print the sources used
        print("\n[Sources used:]")
        for doc in response['source_documents']:
            print(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")

if __name__ == "__main__":
    start_chat()