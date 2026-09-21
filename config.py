import os
from dotenv import load_dotenv

load_dotenv()

# File Paths
DATA_DIRECTORY = "./data"
DB_DIRECTORY = "./chroma_db1"

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_REPO_ID = "Qwen/Qwen2.5-72B-Instruct"