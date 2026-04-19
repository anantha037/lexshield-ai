import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from cv.pipeline import extract_text_from_image, process_pdf

app = FastAPI(title="LexShield AI API", description="Core API for Legal RAG and Intelligence",version="1.0")

# Define a simple data model for incoming queries
class QueryRequest(BaseModel):
    query: str

@app.get("/api/health")
async def health_check():
    """Returns the system status."""
    return {"status":"healthy","service":"LexShield API"}

@app.post("/api/query")
async def process_query(request: QueryRequest):
    """Placeholder for the main RAG query endpoint."""
    return {
        "status":"success",
        "query_received":request.query,
        "message":"RAG pipeline will be integrated here."
    }

@app.post("/api/document")
async def process_document(file: UploadFile = File(...)):
    """Handles document upload, runs OCR, and returns extracted text."""
    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".pdf"):
            extracted_text = process_pdf(temp_file_path)
        elif filename_lower.endswith((".png",".jpg",".jpeg")):
            extracted_text = extract_text_from_image(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, PNG, JPG, or JPEG.")
    
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            "status":"SUCCESS",
            "filename":file.filename,
            "extracted_text":extracted_text
        }
    
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f'Error processing document: {str(e)}')
    