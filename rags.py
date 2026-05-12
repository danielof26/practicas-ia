import os
import csv
import json
import statistics
import time
import spacy
from rag_engine import setup_rag, run_rag

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
THEME         = "f1"
MODEL_NAME    = "gemma3:12b"
EMBED_MODEL   = "snowflake-arctic-embed2"
N_EXEC        = 5
CHUNK_SIZE    = 256
CHUNK_OVERLAP = 50
TOP_K         = 15
DOCS_FOLDER   = "./docs"
SOURCE_FILE = "source_text_f1.txt"
#SOURCE_FILE   = "20042026_BOE-A-2026-8283.pdf"

TEXT_FILE       = os.path.join(DOCS_FOLDER, f"source_doc/{THEME}/{SOURCE_FILE}")
QUESTIONS_CSV   = os.path.join(DOCS_FOLDER, f"source_doc/{THEME}/questions_{THEME}.csv")
CHROMA_PATH     = os.path.join(DOCS_FOLDER, f"chromadb_{THEME}")
OUTPUT_CSV      = os.path.join(DOCS_FOLDER, f"results/{THEME}/validation_{THEME}_{MODEL_NAME}.csv")
EXPERIMENTS_CSV = os.path.join(DOCS_FOLDER, "results/experiments.csv")

PROMPTS = {
    "leyes": """Eres un asistente jurídico especializado en síntesis normativa. 
        Tu objetivo es extraer información de forma estructurada, técnica y extremadamente concisa.
        INSTRUCCIONES DE RESPUESTA:
        1. Prioriza datos cuantitativos (porcentajes %, plazos y cuantías en €).
        2. Identifica claramente los sujetos (quién realiza la acción o a quién afecta).
        3. Si la información está en tablas de sanciones, relaciónala como: [Condición] -> [Multa].
        4. Usa un tono neutro y directo. Evita introducciones como "El texto dice...".
        /no_think""",
    "f1": """You are an expert F1 historian. Answer using ONLY provided context in English."""
}

SPACY_MODELS = {"leyes": "es_core_news_sm", "f1": "en_core_web_sm"}
CHROMA_COLS  = {"leyes": "leyes_docs", "f1": "f1_docs"}

# ─────────────────────────────────────────────
# FUNCIONES DE VALIDACIÓN
# ─────────────────────────────────────────────
nlp = spacy.load(SPACY_MODELS[THEME])

def semantics(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc]

def validate(rag_answer, keys_string):
    sem_answer = semantics(rag_answer)
    key_list = [k.strip() for k in keys_string.split(',')]
    n_found = 0
    details = []
    for k in key_list:
        sem_k = semantics(k)
        found = any(s in sem_answer for s in sem_k)
        n_found += 1 if found else 0
        details.append(f"{k}={'✅' if found else '❌'}")
    return round(n_found/len(key_list), 2), ", ".join(details)

# ─────────────────────────────────────────────
# LEER PREGUNTAS Y SETUP
# ─────────────────────────────────────────────
questions = []
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        questions.append({"question": row["Question"].strip(), "keywords": row["Keywords"].strip()})

query_engine = setup_rag(
    model_name=MODEL_NAME, embed_model=EMBED_MODEL, text_file=TEXT_FILE,
    chroma_path=CHROMA_PATH, chroma_col=CHROMA_COLS[THEME], prompt=PROMPTS[THEME],
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, top_k=TOP_K,
    base_url='http://156.35.160.77:11434'
)

# ─────────────────────────────────────────────
# BLOQUE DE EJECUCIONES (CRONÓMETRO TOTAL)
# ─────────────────────────────────────────────
scores       = [[] for _ in range(len(questions))]
last_answers = [""] * len(questions)

print(f"\nIniciando {N_EXEC} ejecuciones...")
tiempo_inicio_total = time.time()

for exec in range(N_EXEC):
    print(f"\n--- EJECUCIÓN {exec+1}/{N_EXEC} ---")
    responses = run_rag(query_engine, questions)

    for i, (res, q) in enumerate(zip(responses, questions)):
        score, details = validate(res, q["keywords"])
        scores[i].append(score)
        last_answers[i] = res
        print(f"Q{i+1} Score: {score} | {details}")

tiempo_final_total = round(time.time() - tiempo_inicio_total, 2)

# ─────────────────────────────────────────────
# CÁLCULOS ESTADÍSTICOS (DETALLADOS Y GLOBALES)
# ─────────────────────────────────────────────
results = []
medias_score = []

for i, q in enumerate(questions):
    scores_i = scores[i]
    media_s  = round(sum(scores_i) / len(scores_i), 2)
    medias_score.append(media_s)
    
    results.append({
        "question": q["question"], 
        "keywords": q["keywords"], 
        "last_answer": last_answers[i],
        "media": media_s, 
        "minimal": min(scores_i), 
        "maximal": max(scores_i),
        "var": round(statistics.variance(scores_i), 4) if len(scores_i) > 1 else 0.0,
        "std": round(statistics.stdev(scores_i), 4) if len(scores_i) > 1 else 0.0,
        "scores": str(scores_i)
    })

# Guardar CSV Detallado
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["question", "keywords", "last_answer", "media", "minimal", "maximal", "var", "std", "scores"], delimiter=';')
    writer.writeheader()
    writer.writerows(results)

# ESTADÍSTICAS GLOBALES (Con Var y Std del experimento completo)
global_stats = json.dumps({
    "media_score": round(sum(medias_score) / len(medias_score), 2),
    "max_score": max(medias_score),
    "min_score": min(medias_score),
    "global_var": round(statistics.variance(medias_score), 4) if len(medias_score) > 1 else 0.0,
    "global_std": round(statistics.stdev(medias_score), 4) if len(medias_score) > 1 else 0.0,
    "tiempo_total_ejecucion_seg": tiempo_final_total
})

# ─────────────────────────────────────────────
# PERSISTENCIA EN HISTÓRICO EXPERIMENTOS
# ─────────────────────────────────────────────
fieldnames = ["id", "chunk_size", "chunk_overlap", "top_k", "temperature", "model", "score_f1", "score_leyes"]
rows = []
if os.path.isfile(EXPERIMENTS_CSV):
    with open(EXPERIMENTS_CSV, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter=';'))

found = False
for row in rows:
    if (row["chunk_size"] == str(CHUNK_SIZE) and 
        row["chunk_overlap"] == str(CHUNK_OVERLAP) and 
        row["top_k"] == str(TOP_K) and 
        row["model"] == MODEL_NAME):
            row[f"score_{THEME}"] = global_stats
            found = True
            break

if not found:
    rows.append({
        "id": len(rows) + 1, 
        "chunk_size": CHUNK_SIZE, 
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K, 
        "temperature": 0.1, 
        "model": MODEL_NAME,
        "score_f1": global_stats if THEME == "f1" else "N/A",
        "score_leyes": global_stats if THEME == "leyes" else "N/A"
    })

with open(EXPERIMENTS_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[OK] Todo guardado. Tiempo total: {tiempo_final_total}s")
print(f"Stats Globales: {global_stats}")








question = ""
context = ""
llm = ""
index = ""
missing_info = ""
Ollama = ""


# 1. Generate initial response
response = query_engine.query(question)

# 2. Verify against context
verify_prompt = f"""
Context: {context}
Response: {response}
Is the response fully supported by the context? Answer only YES or NO.
"""
verification = llm.complete(verify_prompt)

# 3. Correct if needed
if "NO" in verification.text.upper():
    corrected = query_engine.query(
        f"The following response contains errors: '{response}'. "
        f"Generate a corrected response based only on the context."
    )



query_engine = index.as_query_engine(
    response_mode="refine",
    similarity_top_k=7
)
response = query_engine.query("What is the penalty for recidivism?")



response = query_engine.query(question)

self_eval_prompt = f"""
Question: {question}
Generated response: {response}
Rate the response from 0 to 10 in terms of accuracy and completeness.
If the score is below 7, indicate what information is missing.
"""
evaluation = llm.complete(self_eval_prompt)

if score < 7:
    response = query_engine.query(question + " " + missing_info)




PROMPT_WITH_RULES = """
You are a specialised legal assistant.
MANDATORY RULES:
1. Answer ONLY with information from the provided context.
2. If the information is not found, respond: "Not found in the document."
3. Always cite the article or section where the information comes from.
4. Do not fabricate numerical data such as percentages or fines.
5. If there is a contradiction in the context, state it explicitly.
"""

llm = Ollama(model=MODEL_NAME, system_prompt=PROMPT_WITH_RULES)



PROMPT_XAI = """
Answer the question using ONLY the provided context.
For each statement you make, indicate in brackets the exact fragment 
of the context it comes from.
Format: "The fine is 1000€ [Source: Article 141, paragraph 2]"
If you cannot cite a statement, do not include it.
"""

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)




from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(similarity_top_k=5),
    memory=memory,
    llm=llm,
    system_prompt="You are a legal assistant. Remember the conversation context."
)

response1 = chat_engine.chat("What does RD-Ley 9/2026 establish?")
response2 = chat_engine.chat("And who does it specifically affect?")




llm = Ollama(
    model=MODEL_NAME,
    temperature=0.7,
    system_prompt="""Use the provided context as a basis to generate 
    a clear and didactic explanation. You may rephrase and synthesise 
    but do not fabricate data."""
)

query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    similarity_top_k=10
)