from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import main

app = FastAPI(title="RAG Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "1"

class ChatResponse(BaseModel):
    category: str
    sentiment: str
    response: str
    context: str

@app.get("/")
async def home():
    return HTMLResponse(content=open("templates/index.html", "r", encoding="utf-8").read())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        main.doc_processor.load_pdf(file_path)
        return {"status": "success", "message": f"Uploaded {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Processing query: {request.query}")
        result = main.run_customer_support(request.query, request.thread_id)
        print(f"Response received: {result}")
        return result
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    files = os.listdir("uploads") if os.path.exists("uploads") else []
    return {"documents": files}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        file_path = f"uploads/{filename}"

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Delete physical file
        os.remove(file_path)

        # Rebuild vector store
        main.doc_processor = main.DocumentProcessor()

        for file in os.listdir("uploads"):
            full_path = os.path.join("uploads", file)

            if file.endswith(".pdf"):
                main.doc_processor.load_pdf(full_path)

            elif file.endswith(".txt"):
                main.doc_processor.load_text(full_path)

        return {
            "status": "success",
            "message": f"Deleted {filename} and rebuilt vector database"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)