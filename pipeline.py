import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader

from config import LLM_REPO_ID, DATA_DIRECTORY
from chunking import setup_parent_child_retriever
from retrieval import build_advanced_retriever

load_dotenv()


class RAGPipeline:
    def __init__(self):
        # 1. Setup Base Retriever
        self.base_retriever = setup_parent_child_retriever()

        # 2. Load documents for keyword BM25 search
        self.docs = []
        if os.path.exists(DATA_DIRECTORY):
            pdf_files = [
                os.path.join(DATA_DIRECTORY, f)
                for f in os.listdir(DATA_DIRECTORY)
                if f.lower().endswith(".pdf")
            ]
            for file in pdf_files:
                try:
                    loader = PyMuPDFLoader(file)
                    self.docs.extend(loader.load())
                except Exception as e:
                    print(f"Warning: Failed to load {file}: {e}")

        # 3. Build Advanced Search Pipeline (BM25 + Vector + FlashRank)
        self.retriever = build_advanced_retriever(self.docs, self.base_retriever)

        # 4. Setup LLM
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            task="conversational",
            temperature=0.1,
            max_new_tokens=512,
            huggingfacehub_api_token=token,
        )
        self.chat_model = ChatHuggingFace(llm=llm)

        prompt = PromptTemplate.from_template("""
You are an expert chef assistant. Use the provided recipe context to answer the user.
If the ingredients aren't in the context, say you don't know.

Context: {context}
Question: {question}
Answer:""")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def answer(self, question: str):
        """Returns tuple of (answer_text, retrieved_documents)."""
        result = self.qa_chain.invoke({"query": question})
        return result["result"], result.get("source_documents", [])
