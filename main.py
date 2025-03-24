from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Debugging logs
        print(f"File uploaded: {file.filename} -> {file_path}")

        # Example: Process PDF (Extract text, chunk, store embeddings)
        pdf_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(pdf_text)
        store_chunks_and_embeddings(chunks)

        return {"status": "success", "message": "PDF processed successfully!", "file_path": file_path}

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {"status": "error", "message": "Failed to process PDF."}
    
class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def query_pdf(request: QueryRequest):
    try:
        # Simulate retrieving relevant text from stored embeddings
        retrieved_text = "This is a sample retrieved text based on the question."
        
        # Simulate generating a response (Replace with actual retrieval logic)
        response = f"Answering: {request.question}\nBased on: {retrieved_text}"

        return {"status": "success", "response": response}

    except Exception as e:
        print(f"Error processing query: {e}")
        return {"status": "error", "message": "Failed to retrieve information."}

# Dummy functions for PDF processing (Replace with actual logic)
def extract_text_from_pdf(file_path):
    return "Extracted text from PDF"

def chunk_text(text):
    return [text[i:i+500] for i in range(0, len(text), 500)]  # Simple chunking

def store_chunks_and_embeddings(chunks):
    print(f"Stored {len(chunks)} chunks with embeddings.")
