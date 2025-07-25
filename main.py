from fastapi import FastAPI, UploadFile, Request
from pypdf import PdfReader
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

index = faiss.IndexFlatL2(1536)
stored_chunks = []  # Guarda los chunks para referencia

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
            return {"error": "No se pudo extraer texto del PDF"}, 400

        chunks = chunk_text(full_text)
        embeddings = []
        for chunk in chunks:
            emb = openai.Embedding.create(input=chunk, model="text-embedding-3-small")["data"][0]["embedding"]
            embeddings.append(emb)
        index.add(np.array(embeddings))
        stored_chunks.extend(chunks)
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            return {"error": "No se recibió la pregunta"}, 400
        if index.ntotal == 0 or not stored_chunks:
            return {"error": "No hay información cargada"}, 400

        q_embedding = openai.Embedding.create(input=question, model="text-embedding-3-small")["data"][0]["embedding"]
        D, I = index.search(np.array([q_embedding]), k=3)
        similar_chunks = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]

        context = "\n---\n".join(similar_chunks)
        response = openai.ChatCompletion.create(
            model="gpt-4-mini",
            messages=[{"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}]
        )
        return {
            "answer": response.choices[0].message.content,
            "context": similar_chunks
        }
    except Exception as e:
        return {"error": str(e)}, 500