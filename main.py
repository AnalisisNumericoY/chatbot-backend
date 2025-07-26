from fastapi import FastAPI, UploadFile, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json

app = FastAPI()

# Configuración CORS (imprescindible)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

index = faiss.IndexFlatL2(1536)
stored_chunks = []

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        reader = PdfReader(file.file)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="El PDF no contiene texto legible")

        chunks = chunk_text(full_text)
        embeddings = []
        for chunk in chunks:
            emb = openai.Embedding.create(
                input=chunk,
                model="text-embedding-3-small"
            )["data"][0]["embedding"]
            embeddings.append(emb)
        
        index.add(np.array(embeddings))
        stored_chunks.extend(chunks)
        return {"status": "success", "message": f"PDF procesado con {len(chunks)} fragmentos"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {str(e)}")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: Request):
    try:
        # Manejo compatible con Streamlit y React
        try:
            data = await request.json()
            question = data.get("question")
        except:
            body = await request.body()
            data = json.loads(body)
            question = data.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="No se recibió pregunta")
        
        if index.ntotal == 0:
            raise HTTPException(status_code=400, detail="Primero carga documentos via /upload")

        q_embedding = openai.Embedding.create(
            input=question,
            model="text-embedding-3-small"
        )["data"][0]["embedding"]

        D, I = index.search(np.array([q_embedding]), k=3)
        similar_chunks = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]
        context = "\n---\n".join(similar_chunks)

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Responde basándote en el contexto"},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
            ],
            temperature=0.7
        )

        return {
            "answer": response.choices[0].message.content,
            "context": similar_chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))