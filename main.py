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



# Configura CORS (¡IMPORTANTE!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Modelo para validación de entrada
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: Request):
    try:
        # Parsear el JSON manualmente
        data = await request.json()
        question = data.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="No se recibió la pregunta")
        
        if index.ntotal == 0 or not stored_chunks:
            raise HTTPException(status_code=400, detail="No hay información cargada")

        # Generar embedding
        q_embedding = openai.Embedding.create(
            input=question, 
            model="text-embedding-3-small"
        )["data"][0]["embedding"]

        # Búsqueda de chunks similares
        D, I = index.search(np.array([q_embedding]), k=3)
        similar_chunks = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]
        context = "\n---\n".join(similar_chunks)

        # Generar respuesta con GPT
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Cambiado de "gpt-4-mini" a "gpt-4"
            messages=[
                {"role": "system", "content": "Responde basándote en el contexto proporcionado."},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
            ]
        )

        return {
            "status": "success",
            "answer": response.choices[0].message.content,
            "context": similar_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))