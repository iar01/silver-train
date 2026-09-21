import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    os.environ["HF_TOKEN"] = token

# File Paths
DATA_DIRECTORY = str(BASE_DIR / "data")
DB_DIRECTORY = str(BASE_DIR / "chroma_db1")

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_REPO_ID = "Qwen/Qwen2.5-72B-Instruct"