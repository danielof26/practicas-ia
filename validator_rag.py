import os
import csv
import spacy
from rag_engine import setup_rag, run_rag

THEME = "f1"
MODEL_NAME = "llama3.2"
EMBED_MODEL = "snowflake-arctic-embed2"
N_EXEC = 5
DOCS_FOLDER = "./docs"

TEXT_FILE     = os.path.join(DOCS_FOLDER, f"source_doc/{THEME}/source_text_{THEME}.txt")
QUESTIONS_CSV = os.path.join(DOCS_FOLDER, f"source_doc/{THEME}/questions_{THEME}.csv")
CHROMA_PATH   = os.path.join(DOCS_FOLDER, f"chromadb_{THEME}")
OUTPUT_CSV    = os.path.join(DOCS_FOLDER, f"results/{THEME}/validation_{THEME}_{MODEL_NAME}.csv")


PROMPTS = {
    "leyes": """Eres un asistente jurídico especializado en síntesis normativa. 
        Tu objetivo es extraer información de forma estructurada, técnica y extremadamente concisa.

        INSTRUCCIONES DE RESPUESTA:
        1. Prioriza datos cuantitativos (porcentajes %, plazos y cuantías en €).
        2. Identifica claramente los sujetos (quién realiza la acción o a quién afecta).
        3. Si la información está en tablas de sanciones, relaciónala como: [Condición] -> [Multa].
        4. Usa un tono neutro y directo. Evita introducciones como "El texto dice...".
        5. Si una pregunta implica un "Cuándo", busca el umbral numérico que activa la norma.""",
    
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
    
    return round(n_found/len(key_list),2), ", ".join(details)



questions = []
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        questions.append({
            "question": row["Question"].strip(),
            "keywords": row["Keywords"].strip()
        })



query_engine = setup_rag(
    model_name   = MODEL_NAME,
    embed_model  = EMBED_MODEL,
    text_file    = TEXT_FILE,
    chroma_path  = CHROMA_PATH,
    chroma_col   = CHROMA_COLS[THEME],
    prompt       = PROMPTS[THEME],
    chunk_size   = 1024,
    chunk_overlap= 200,
    top_k        = 15
)


scores = [[] for _ in range(len(questions))] # list of length N_QUESTIONS of lists with length N_EXEC
last_answers = [""] * len(questions) #stores only the last response of each of the questions


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



results = []
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")


for i, q in enumerate(questions):
    scores_i = scores[i]
    media = round(sum(scores_i)/len(scores_i),2)
    minimal = min(scores_i)
    maximal = max(scores_i)

    print(f"[{i+1}] Media: {media} | Min: {minimal} | Max: {maximal}")
    print(f"  Q: {q['question']}")
    print(f"  Scores por ejecución: {scores_i}\n")

    results.append({
        "question":    q["question"],
        "keywords":    q["keywords"],
        "last_answer": last_answers[i],
        "media":       media,
        "minimal":      minimal,
        "maximal":      maximal,
        "scores":      str(scores)
    })