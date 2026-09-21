import json
import os
import sys
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# Add project root to sys.path so pipeline and config can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline import RAGPipeline
from config import EMBEDDING_MODEL

def run_evaluation():
    # 1. Load test dataset
    dataset_path = Path(__file__).resolve().parent / "test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    rag_system = RAGPipeline()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 2. Query your RAG pipeline
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        # Get answer & retrieved docs from pipeline
        response, retrieved_docs = rag_system.answer(q)
        
        questions.append(q)
        answers.append(response)
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(gt)

    # 3. Format as HuggingFace Dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # 4. Compute metrics using RAGAS
    eval_kwargs = {}
    if not os.getenv("OPENAI_API_KEY"):
        eval_kwargs["llm"] = LangchainLLMWrapper(rag_system.chat_model)
        eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        **eval_kwargs
    )

    print("\n=== Evaluation Results ===")
    print(results)

if __name__ == "__main__":
    run_evaluation()