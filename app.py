from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from pipeline import RAGPipeline

from config import DATA_DIRECTORY

app = FastAPI(title="Recipe RAG Assistant API")

# Initialize pipeline instance
rag_system = RAGPipeline()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Recipe RAG Service is running"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        answer, sources = rag_system.answer(request.question)
        source_texts = [doc.page_content for doc in sources]
        return QueryResponse(answer=answer, sources=source_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        save_dir = os.path.join(DATA_DIRECTORY, "recipe_pdfs")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Re-index newly uploaded document
        rag_system.reload_documents(file_path)
        return {"message": f"Successfully uploaded and indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))