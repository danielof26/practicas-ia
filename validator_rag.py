import os
import csv
import json
import statistics
import spacy
from rag_engine import setup_rag, run_rag

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
THEME         = "leyes"
MODEL_NAME    = "qwen3:4b"
EMBED_MODEL   = "snowflake-arctic-embed2"
N_EXEC        = 5
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 3
DOCS_FOLDER   = "./docs"
#SOURCE_FILE   = "source_text_f1.txt"
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
        5. Si una pregunta implica un "Cuándo", busca el umbral numérico que activa la norma.
        /no_think""",
    "f1": """You are an expert F1 historian.
        Answer using ONLY the provided context.
        Answer always in English. Be concise and precise."""
}

SPACY_MODELS = {
    "leyes": "es_core_news_sm",
    "f1":    "en_core_web_sm"
}

CHROMA_COLS = {
    "leyes": "leyes_docs",
    "f1":    "f1_docs"
}

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
# LEER PREGUNTAS
# ─────────────────────────────────────────────
questions = []
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        questions.append({
            "question": row["Question"].strip(),
            "keywords": row["Keywords"].strip()
        })

# ─────────────────────────────────────────────
# SETUP RAG
# ─────────────────────────────────────────────
query_engine = setup_rag(
    model_name    = MODEL_NAME,
    embed_model   = EMBED_MODEL,
    text_file     = TEXT_FILE,
    chroma_path   = CHROMA_PATH,
    chroma_col    = CHROMA_COLS[THEME],
    prompt        = PROMPTS[THEME],
    chunk_size    = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    top_k         = TOP_K,
    base_url      = 'http://156.35.95.18:11434'
)

# ─────────────────────────────────────────────
# N EJECUCIONES
# ─────────────────────────────────────────────
scores       = [[] for _ in range(len(questions))]
last_answers = [""] * len(questions)

for exec in range(N_EXEC):
    print(f"\n{'='*50}")
    print(f"EXEC_ {exec+1}/{N_EXEC}")
    print(f"{'='*50}")
    responses = run_rag(query_engine, questions)
    for i, (res, q) in enumerate(zip(responses, questions)):
        score, details = validate(res, q["keywords"])
        scores[i].append(score)
        last_answers[i] = res
        print(f"Score {score} | {details}")

# ─────────────────────────────────────────────
# RESULTADOS POR PREGUNTA
# ─────────────────────────────────────────────
results = []
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")

for i, q in enumerate(questions):
    scores_i = scores[i]
    media   = round(sum(scores_i) / len(scores_i), 2)
    minimal = min(scores_i)
    maximal = max(scores_i)
    # varianza y desviación requieren al menos 2 valores
    var = round(statistics.variance(scores_i), 4) if len(scores_i) > 1 else 0.0
    std = round(statistics.stdev(scores_i), 4)    if len(scores_i) > 1 else 0.0

    print(f"[{i+1}] Media: {media} | Min: {minimal} | Max: {maximal} | Var: {var} | Std: {std}")
    print(f"  Q: {q['question']}")
    print(f"  Scores por ejecución: {scores_i}\n")

    results.append({
        "question":    q["question"],
        "keywords":    q["keywords"],
        "last_answer": last_answers[i],
        "media":       media,
        "minimal":     minimal,
        "maximal":     maximal,
        "var":         var,
        "std":         std,
        "scores":      str(scores_i)
    })

# Guarda CSV de detalle por pregunta
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["question", "keywords", "last_answer", "media", "minimal", "maximal", "var", "std", "scores"],
        delimiter=';'
    )
    writer.writeheader()
    writer.writerows(results)

# ─────────────────────────────────────────────
# ESTADÍSTICAS GLOBALES
# ─────────────────────────────────────────────
medias       = [round(sum(scores[i]) / len(scores[i]), 2) for i in range(len(questions))]
global_media = round(sum(medias) / len(medias), 2)
global_max   = round(max(medias), 2)
global_min   = round(min(medias), 2)
global_var   = round(statistics.variance(medias), 4) if len(medias) > 1 else 0.0
global_std   = round(statistics.stdev(medias), 4)    if len(medias) > 1 else 0.0

global_stats = json.dumps({
    "media": global_media,
    "max":   global_max,
    "min":   global_min,
    "var":   global_var,
    "std":   global_std
})

print(f"\nScore global del experimento: {global_stats}")

# ─────────────────────────────────────────────
# GUARDAR EN EXPERIMENTS CSV
# Si existe fila con misma configuración, actualiza score del dataset
# Si no existe, crea fila nueva
# ─────────────────────────────────────────────
fieldnames = ["id", "chunk_size", "chunk_overlap", "top_k", "temperature", "model", "score_f1", "score_leyes"]

rows = []
file_exists = os.path.isfile(EXPERIMENTS_CSV)
if file_exists:
    with open(EXPERIMENTS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

found = False
for row in rows:
    if (row["chunk_size"]    == str(CHUNK_SIZE) and
        row["chunk_overlap"] == str(CHUNK_OVERLAP) and
        row["top_k"]         == str(TOP_K) and
        row["temperature"]   == str(0.1) and
        row["model"]         == MODEL_NAME):
        if THEME == "f1":
            row["score_f1"] = global_stats
        else:
            row["score_leyes"] = global_stats
        found = True
        break

if not found:
    rows.append({
        "id":            len(rows) + 1,
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k":         TOP_K,
        "temperature":   0.1,
        "model":         MODEL_NAME,
        "score_f1":      global_stats if THEME == "f1" else "N/A",
        "score_leyes":   global_stats if THEME == "leyes" else "N/A"
    })

with open(EXPERIMENTS_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(rows)

print(f"Experimento guardado en: {EXPERIMENTS_CSV}")