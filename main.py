# main.py
from fastapi import FastAPI, UploadFile
from pypdf import PdfReader
import openai
import faiss
import numpy as np

app = FastAPI()

# Configura FAISS y OpenAI
index = faiss.IndexFlatL2(1536)  # Dimensión de text-embedding-3-small
openai.api_key = "tu-api-key"

@app.post("/upload")
async def upload_file(file: UploadFile):
    text = PdfReader(file.file).pages[0].extract_text()
    embedding = openai.Embedding.create(input=text, model="text-embedding-3-small")["data"][0]["embedding"]
    index.add(np.array([embedding]))  # Guardar en FAISS
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(question: str):
    q_embedding = openai.Embedding.create(input=question, model="text-embedding-3-small")["data"][0]["embedding"]
    _, similar_chunks = index.search(np.array([q_embedding]), k=3)  # Buscar en FAISS
    response = openai.ChatCompletion.create(
        model="gpt-4-mini",
        messages=[{"role": "user", "content": f"Contexto: {similar_chunks}\nPregunta: {question}"}]
    )
    return {"answer": response.choices[0].message.content}

