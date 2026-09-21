# 🍳 Advanced Recipe RAG Chatbot

An intelligent, context-aware Recipe Assistant powered by **Advanced Retrieval-Augmented Generation (RAG)**. This chatbot indexes culinary cookbooks and recipe guides in PDF format, performs hybrid retrieval combining keyword matching (BM25) and dense semantic vector search, refines results with a cross-encoder re-ranker (FlashRank), and generates chef-grade answers using Hugging Face's state-of-the-art **Qwen2.5-72B-Instruct** model.

It also features a full evaluation suite using **Ragas** to assess faithfulness, answer relevancy, and context retrieval quality.

---

## 🌟 Key Features

- **High-Performance Document Ingestion**: Fast extraction from multi-page recipe PDFs using `PyMuPDF`.
- **Parent-Child Chunking**: Splits documents into small child chunks for precise semantic matching while returning complete parent recipes to the LLM for rich context.
- **Hybrid Retrieval Pipeline**: Combines **BM25 keyword search** (for exact ingredient and dish names) with **dense vector semantic search** via `EnsembleRetriever`.
- **Cross-Encoder Re-Ranking**: Employs **FlashRank** (`FlashrankRerank`) to score and re-order candidate passages for optimal relevance.
- **State-of-the-Art LLM**: Connects to `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Inference API for accurate, conversational culinary instructions.
- **RAG Evaluation Suite (Ragas)**: Built-in automated benchmarking for `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall` against golden datasets.

---

## 🏗️ Architecture

```
                       User Question
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  BM25 Keyword    │          │  ChromaDB Dense  │
    │  Retriever       │          │  Vector Search   │
    │  (Top-k Exact)   │          │  (Parent-Child)  │
    └─────────┬────────┘          └─────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
              ┌─────────────────────────────┐
              │      Ensemble Retriever     │
              │  (Weighted Candidate Pool)  │
              └──────────────┬──────────────┘
                             ▼
              ┌─────────────────────────────┐
              │      FlashRank Reranker     │
              │   (Cross-Encoder Scoring)   │
              └──────────────┬──────────────┘
                             ▼
              ┌─────────────────────────────┐
              │    Qwen2.5-72B-Instruct     │
              │ (Hugging Face Inference)    │
              └──────────────┬──────────────┘
                             ▼
                      Chef Bot Answer
```

---

## 📁 Project Structure

```text
rag-chat-bot/
├── data/                  # Recipe PDF documents (knowledge base)
├── chroma_db1/            # Local Chroma vector store (created during ingestion)
├── eval/
│   ├── evaluate.py        # Automated RAG evaluation using Ragas
│   ├── test_dataset.json  # Benchmark questions and ground-truth references
│   └── __init__.py
├── chunking.py            # Parent-Document Retriever setup & chunk splitters
├── config.py              # Centralized environment variables, model IDs & directory paths
├── database.py            # ChromaDB client & vector store initialization
├── ingest.py              # Ingestion pipeline for processing PDFs into ChromaDB
├── main.py                # Interactive CLI chef chatbot interface
├── pipeline.py            # Modular RAG pipeline (retrieval + QA chain)
├── retrieval.py           # Hybrid retrieval (BM25 + Vector) and FlashRank reranking
├── setup_db.py            # Standalone database ingestion utility
├── requirements.txt       # Project dependencies
├── .env.example           # Template for environment variables
└── .gitignore             # Git exclusion rules
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- A free [Hugging Face Account](https://huggingface.co/) and an [Access Token](https://huggingface.co/settings/tokens)

### 2. Clone the Repository

```bash
git clone https://github.com/iar01/silver-train.git
cd silver-train
```

### 3. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory (based on `.env.example`):

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
```

*(Optional: Set `OPENAI_API_KEY=your_key` if you wish to run Ragas evaluations using OpenAI as the evaluator model).*

---

## 📖 Usage

### Step 1: Ingest Recipe Data

Place your PDF recipe books into the `data/` folder and run:

```bash
python ingest.py
```

This parses the PDFs with `PyMuPDF` and builds the parent-child vector index inside `./chroma_db1`.

### Step 2: Run the Chatbot

Start the interactive chef assistant:

```bash
python main.py
```

Ask questions such as:
- *"How do I make traditional hummus from scratch?"*
- *"What ingredients do I need for garlic sauce?"*
- *"Can you give me a healthy Middle Eastern dinner recipe?"*

Type `exit` or `quit` to end the session.

### Step 3: Run RAG Evaluation

Evaluate the retrieval and generation performance using Ragas:

```bash
python eval/evaluate.py
```

Or from inside the `eval/` folder:

```bash
cd eval
python evaluate.py
```

This runs test queries from `eval/test_dataset.json` through the pipeline and prints metric scores for:
- **Faithfulness**: Measures if the answer is grounded solely in the retrieved context.
- **Answer Relevancy**: Evaluates how pertinent the response is to the user's question.
- **Context Precision**: Assesses whether relevant chunks were ranked at the top.
- **Context Recall**: Checks whether all relevant ground-truth information was retrieved.

---

## ⚙️ Configuration & Customization

All core settings are centralized in `config.py`:

- **LLM Model**: Configured to `Qwen/Qwen2.5-72B-Instruct` via `LLM_REPO_ID`.
- **Embeddings**: Configured to `all-MiniLM-L6-v2` via `EMBEDDING_MODEL`.
- **Directories**: Paths for `data/` and `chroma_db1/` are dynamically resolved.
- **Chunk Sizes**: Adjust chunk and overlap sizes in `chunking.py`.
- **Retrieval Weights & Top-N**: Tune BM25 vs. Vector balance and reranker `top_n` in `retrieval.py`.
