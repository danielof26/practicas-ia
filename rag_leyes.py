import csv
import os
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PyMuPDFReader # Lector especializado
import chromadb
import logging

# --- CONFIGURACIÓN DE MODELOS ---
MODEL_NAME = "qwen3:4b"
EMBED_MODEL = "snowflake-arctic-embed2"
BASE_URL = "http://156.35.95.18:11434"

# --- RUTAS ---
DOCS_FOLDER = "/Users/danielonisfabian/Desktop/GRUPO_INVESTIGACION/docs"
PDF_SOURCE = os.path.join(DOCS_FOLDER, "source_doc/leyes/20042026_BOE-A-2026-8283.pdf") 
QUESTIONS_CSV = os.path.join(DOCS_FOLDER, "source_doc/leyes/questions_leyes.csv")
OUTPUT_CSV = os.path.join(DOCS_FOLDER, f"results/leyes/rag_answers_{MODEL_NAME}.csv")
CHROMA_PATH = os.path.join(DOCS_FOLDER, "chromadb_leyes")


PROMPT = """Eres un experto jurídico implacable. Tu tarea es analizar el Real Decreto-ley 9/2026. 

INSTRUCCIONES CRÍTICAS:
1. Si la pregunta menciona sanciones o multas, BUSCA exhaustivamente tablas o listas con cantidades en euros (€).
2. Es obligatorio identificar las infracciones como 'GRAVES' o 'MUY GRAVES' si aparecen.
3. Utiliza términos técnicos: 'ineludible', 'desglosada en factura', 'coeficiente dinámico'.
4. Si la respuesta implica cifras (ej. 401€, 601€, 2001€), inclúyelas sin falta.
Responde de forma concisa y técnica en español."""


# --- INICIALIZACIÓN ---
llm = Ollama(
    model=MODEL_NAME,
    request_timeout=600.0, 
    system_prompt=PROMPT,
    context_window=8000,
    base_url=BASE_URL,
    temperature=0.1
)

Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
Settings.llm = llm
# Configuramos trozos grandes para no romper los artículos de la ley
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)


db = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = db.get_or_create_collection("leyes_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


if chroma_collection.count() == 0:
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path=PDF_SOURCE)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )


query_engine = index.as_query_engine(similarity_top_k=25)


questions = []
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        questions.append(row)


results = []
for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] {q['Question']}")
    response = query_engine.query(q["Question"])
    rag_answer = str(response).strip().replace('\n', ' ').replace(';', ',')
    
    print(f"  RAG: {rag_answer}\n")
    results.append({
        "question": q["Question"],
        "rag_answer": rag_answer
    })

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["question", "rag_answer"], delimiter=';')
    writer.writeheader()
    writer.writerows(results)