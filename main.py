from fastapi import FastAPI, UploadFile, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI  # Nueva importación
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Nueva inicialización

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
        # Verificar tipo de archivo
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
        
        reader = PdfReader(file.file)
        full_text = ""
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            except:
                continue
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF. ¿Está escaneado?")

        chunks = chunk_text(full_text)
        embeddings = []
        for chunk in chunks:
            emb = client.embeddings.create(  # Llamada actualizada
                input=chunk,
                model="text-embedding-3-small"
            ).data[0].embedding
            embeddings.append(emb)
        
        index.add(np.array(embeddings))
        stored_chunks.extend(chunks)
        return {"status": "success", "message": f"PDF procesado. {len(chunks)} fragmentos almacenados."}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {str(e)}")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: Request):
    try:
        # Compatibilidad con diferentes formatos de solicitud
        try:
            data = await request.json()
        except:
            body = await request.body()
            data = json.loads(body)
        
        question = data.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="No se recibió pregunta")
        
        if index.ntotal == 0:
            raise HTTPException(status_code=400, detail="Primero carga documentos PDF via /upload")

        # Generar embedding
        q_embedding = client.embeddings.create(  # Llamada actualizada
            input=question,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Búsqueda de chunks similares
        D, I = index.search(np.array([q_embedding]), k=3)
        similar_chunks = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]
        context = "\n---\n".join(similar_chunks)

        # Generar respuesta
        response = client.chat.completions.create(  # Llamada actualizada
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Responde basándote en el contexto proporcionado."},
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