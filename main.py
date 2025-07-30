from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI  # Nueva importación
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json
from bs4 import BeautifulSoup
from pptx import Presentation
import httpx
import logging

app = FastAPI()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Persistencia con Faiss y JSON ---
FAISS_INDEX_PATH = "vectorstore.faiss"
CHUNKS_PATH = "chunks.json"

# Cargar índice y chunks si existen
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(1536)  # Dimensión de text-embedding-3-small

if os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        stored_chunks = json.load(f)
else:
    stored_chunks = []
# --- Fin de la configuración de persistencia ---


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_store_text(text: str):
    """Chunks text, creates embeddings, and stores them."""
    if not text or not text.strip():
        return 0
    
    chunks = chunk_text(text)
    
    if not chunks:
        return 0

    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding
        embeddings.append(emb)
    
    if embeddings:
        index.add(np.array(embeddings))
        stored_chunks.extend(chunks)

        # Guardar cambios en disco
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(stored_chunks, f, ensure_ascii=False, indent=2)
    
    return len(chunks)

@app.post("/upload_pdf")
async def upload_file(file: UploadFile = File(...)):
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
            raise HTTPException(
            status_code=400,
            detail="No pude leer el texto de este PDF. Por favor verifica que: \n1. El PDF contenga texto seleccionable (no sea una imagen escaneada)\n2. No esté protegido con contraseña\n3. Tenga al menos un párrafo de texto"
            )

        num_chunks = process_and_store_text(full_text)
        
        if num_chunks == 0:
            raise HTTPException(status_code=500, detail="No se pudieron procesar fragmentos del documento.")

        return {
    "status": "success",
    "message": f"PDF procesado. {num_chunks} fragmentos almacenados.",
    "full_text": full_text  # ✅ importante
}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {str(e)}")

@app.post("/upload_pptx")
async def upload_pptx(file: UploadFile):
    try:
        if not file.filename.endswith(('.pptx')):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos .pptx")
        
        full_text = ""
        presentation = Presentation(file.file)
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    full_text += shape.text + "\n"

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No se encontró texto en el archivo PPTX.")

        num_chunks = process_and_store_text(full_text)

        if num_chunks == 0:
            raise HTTPException(status_code=500, detail="No se pudieron procesar fragmentos del documento.")

        return {
    "status": "success",
    "message": f"PPTX procesado. {num_chunks} fragmentos almacenados.",
    "full_text": full_text
}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar PPTX: {str(e)}")

class UrlRequest(BaseModel):
    url: str

@app.post("/scrape_url")
async def scrape_url(request: UrlRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.url, follow_redirects=True, timeout=20.0)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        full_text = soup.get_text()
        
        lines = (line.strip() for line in full_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        full_text = '\n'.join(chunk for chunk in chunks if chunk)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No se encontró texto procesable en la URL.")

        num_chunks = process_and_store_text(full_text)
        
        if num_chunks == 0:
            raise HTTPException(status_code=500, detail="No se pudieron procesar fragmentos del contenido de la URL.")

        return {"status": "success", "message": f"URL procesada. {num_chunks} fragmentos almacenados."}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Error al acceder a la URL: {e.response.status_code} {e.request.url}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error de red al intentar acceder a la URL: {e.request.url}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la URL: {str(e)}")

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
            logger.info("Solicitud a /ask sin pregunta.")
            return {
                "answer": "Parece que no se ha enviado ninguna pregunta. Por favor, inténtalo de nuevo.",
                "context": []
            }
        
        if index.ntotal == 0:
            logger.info("Solicitud a /ask con índice vacío.")
            return {
                "answer": "Actualmente no tengo documentos con los que trabajar. Por favor, pide a un administrador que suba la información necesaria (PDF, PPTX, etc.) para que pueda responder a tus preguntas.",
                "context": []
            }
            
        # Generar embedding
        q_embedding = client.embeddings.create(  # Llamada actualizada
            input=question,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Búsqueda de chunks similares
        _, I = index.search(np.array([q_embedding]), k=3)
        similar_chunks = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]
        context = "\n---\n".join(similar_chunks)

        # Generar respuesta
        system_prompt = """Eres un asistente virtual de atención al cliente. Tu misión es ayudar a los usuarios de forma amable y profesional.
1.  **Analiza el contexto**: Basa tu respuesta únicamente en el fragmento de texto proporcionado en el contexto. No utilices conocimiento externo ni inventes información.
2.  **Tono**: Mantén un tono corporativo pero cercano y amigable. Usa un lenguaje claro y evita la jerga técnica si es posible.
3.  **Si encuentras la respuesta**: Proporciónala de manera completa y fácil de entender.
4.  **Si NO encuentras la respuesta**: Indica amablemente que la información no se encuentra en los documentos disponibles. Ofrece alternativas, como contactar a un agente humano o revisar la pregunta.
5.  **Cierre proactivo**: Finaliza siempre tus respuestas de forma positiva, preguntando si hay algo más en lo que puedas ayudar. Por ejemplo: '¿Hay algo más en lo que pueda asistirte hoy?'"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
            ],
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "context": similar_chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
