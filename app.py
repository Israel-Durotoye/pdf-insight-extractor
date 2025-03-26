from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional

# Importing necessary functions
from pdf_extractor import extract_text_from_pdf
from embedder import chunk_text, store_chunks_and_embeddings, hybrid_search
from query_llm import generate_response

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store document mappings
document_store = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_location = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(file_location, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_location)
        
        # Chunk, embed, and store in ChromaDB
        chunks = chunk_text(extracted_text)
        store_chunks_and_embeddings(chunks)
        
        document_id = file.filename
        document_store[document_id] = chunks
        
        return {
            "success": True,
            "filename": file.filename,
            "message": "PDF uploaded, processed, and embedded successfully",
            "document_id": document_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

@app.post("/query/")
async def query_document(query_request: QueryRequest):
    try:
        if query_request.document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document ID not found")
        
        # Retrieve relevant chunks using hybrid search
        retrieved_chunks = hybrid_search(query_request.question, top_k=5)
        
        # Generate response using LLM
        answer = generate_response(query_request.question, retrieved_chunks)
        
        return {
            "success": True,
            "answer": answer,
            "source": "Extracted text from document"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
