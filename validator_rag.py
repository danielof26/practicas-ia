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
THEME         = "leyes"
MODEL_NAME    = "qwen3:8b"
EMBED_MODEL   = "snowflake-arctic-embed2"
N_EXEC        = 5
CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 200
TOP_K         = 15
DOCS_FOLDER   = "./docs"
#SOURCE_FILE = "source_text_f1.txt"
SOURCE_FILE   = "20042026_BOE-A-2026-8283.pdf"

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