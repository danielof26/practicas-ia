import csv
import os
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
 

MODEL_NAME = "qwen3.5:4b"
EMBED_MODEL = "snowflake-arctic-embed2"


DOCS_FOLDER = "/Users/danielonisfabian/Desktop/GRUPO_INVESTIGACION/docs"
TEXT_FILE = os.path.join(DOCS_FOLDER, "source_doc/source_text_f1.txt")
QUESTIONS_CSV = os.path.join(DOCS_FOLDER, "source_doc/questions_f1.csv")
OUTPUT_CSV = os.path.join(DOCS_FOLDER, f"results/rag_answers_{MODEL_NAME}.csv")
CHROMA_PATH = os.path.join(DOCS_FOLDER, "chromadb_f1")


llm = Ollama(
    model = MODEL_NAME,
    request_timeout=600.0, 
    system_prompt='You are an expert F1 historian. Answer using ONLY the provided context. Answer always in English. Be concise and precise',
    context_window=8000,
    temperature=0.1
)

# Module configuration
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
Settings.llm = llm


# Creates DB, opens collection f1_docs

db = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = db.get_or_create_collection("f1_docs") #collection f1_docs
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
query_engine = index.as_query_engine(similarity_top_k=5)  


questions = []  #vector for storing the questions

# stores questions in list
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        questions.append({
            "question": row["Question"].strip(),
            "answer": row["Answer"].strip(),
            "keywords": row["Keywords"].strip(),
            "category": row["Category"].strip()
        })


results = []  #vector for storing the answers made by RAG

for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] [{q['category']}] {q['question']}")

    response = query_engine.query(q["question"]) # sends the question to RAG, converts it to vector, search for 5 more similar and sends them alongside the question
    rag_answer = str(response).strip()   # converts answer to string

    print(f"  RAG: {rag_answer}\n")
    
    results.append({  #stores the results of the RAG
        "category": q["category"],
        "question": q["question"],
        "rag_answer": rag_answer
    })


with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["category", "question", "rag_answer"], delimiter=';')
    writer.writeheader()
    writer.writerows(results)




