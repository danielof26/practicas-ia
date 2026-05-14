import csv
import os
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
 

MODEL_NAME = "qwen3:4b"
EMBED_MODEL = "snowflake-arctic-embed2"


TEXT_SOURCE = "leyes/source_text_leyes.txt"
CSV_SOURCE = "leyes/questions_leyes.csv"
CHROMADB = "chromadb_leyes"
CHROMADB_COL = "leyes_docs"
PROMPT = """Eres un experto jurídico implacable. 
Tu tarea es analizar el Real Decreto-ley 9/2026. 

INSTRUCCIONES CRÍTICAS:
1. Si la pregunta menciona sanciones o multas, BUSCA en todo el contexto proporcionado cantidades en euros (€), infracciones 'graves' o 'muy graves'.
2. PROHIBIDO responder "no hay información" si el contexto menciona la Ley 16/1987 o cuantías económicas.
3. Utiliza términos técnicos: 'ineludible', 'desglosada en factura', 'coeficiente dinámico'.
4. Si encuentras cifras (ej. 401€, 601€, 2001€), inclúyelas OBLIGATORIAMENTE.
Responde de forma concisa y técnica en español."""


DOCS_FOLDER = "/Users/danielonisfabian/Desktop/GRUPO_INVESTIGACION/docs"
TEXT_FILE = os.path.join(DOCS_FOLDER, f"source_doc/{TEXT_SOURCE}")
QUESTIONS_CSV = os.path.join(DOCS_FOLDER, f"source_doc/{CSV_SOURCE}")
OUTPUT_CSV = os.path.join(DOCS_FOLDER, f"results/leyes/rag_answers_{MODEL_NAME}.csv")
CHROMA_PATH = os.path.join(DOCS_FOLDER, CHROMADB)


llm = Ollama(
    model = MODEL_NAME,
    request_timeout=600.0, 
    system_prompt=PROMPT,
    context_window=8000,
    base_url="http://156.35.95.18:11434",
    temperature=0.1
)

# Module configuration
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
Settings.llm = llm


# Creates DB, opens collection f1_docs

db = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = db.get_or_create_collection(CHROMADB_COL) #collection f1_docs
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() == 0:    # first time exceuted
    documents = SimpleDirectoryReader(input_files=[TEXT_FILE]).load_data() #loads source_text_f1 in memory
    index = VectorStoreIndex.from_documents(    #divides in chucks, generates embeddings and store in chromadb
        documents,
        storage_context=storage_context
    )
else:
    index = VectorStoreIndex.from_vector_store(  #loads vectors
        vector_store,
        storage_context=storage_context
    )


#creates RAG query engine that retrieves the 5 most semantic similar vectors
query_engine = index.as_query_engine(similarity_top_k=15)  


questions = []  #vector for storing the questions

# stores questions in list
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        questions.append({
            "question": row["Question"].strip(),
            "answer": row["Answer"].strip(),
            "keywords": row["Keywords"].strip(),
            #"category": row["Category"].strip()
        })


results = []  #vector for storing the answers made by RAG

for i, q in enumerate(questions):
    #print(f"[{i+1}/{len(questions)}] [{q['category']}] {q['question']}")
    print(f"[{i+1}/{len(questions)}] {q['question']}")

    response = query_engine.query(q["question"]) # sends the question to RAG, converts it to vector, search for 5 more similar and sends them alongside the question
    rag_answer = str(response).strip().replace('\n', ' ').replace(';', ',')   # converts answer to string

    print(f"  RAG: {rag_answer}\n")
    
    results.append({  #stores the results of the RAG
        #"category": q["category"],
        "question": q["question"],
        "rag_answer": rag_answer
    })


with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    #writer = csv.DictWriter(f, fieldnames=["category", "question", "rag_answer"], delimiter=';')
    writer = csv.DictWriter(f, fieldnames=["question", "rag_answer"], delimiter=';')

    writer.writeheader()
    writer.writerows(results)




