import os
import csv
from llama_index.llms.ollama import Ollama

# --- CONFIGURACIÓN ---
JUDGE_NAME = "gemma3:12b"  # El modelo que actuará como juez
MODEL_NAME = "qwen3:4b"  # El modelo que actuará como juez
TEMA = "leyes"
BASE_URL = "http://156.35.95.18:11434"

DOCS_FOLDER = "/Users/danielonisfabian/Desktop/GRUPO_INVESTIGACION/docs"
QUESTIONS_CSV = os.path.join(DOCS_FOLDER, f"source_doc/{TEMA}/questions_{TEMA}.csv")
OUTPUT_CSV = os.path.join(DOCS_FOLDER, f"results/{TEMA}/rag_answers_{MODEL_NAME}.csv")

# Inicializamos al Juez
llm_judge = Ollama(
    model=JUDGE_NAME, 
    base_url=BASE_URL, 
    request_timeout=120.0,
    temperature=0.0 # Temperatura 0 para que sea un juez estricto y consistente
)

def evaluar_con_ia(pregunta, respuesta_esperada, respuesta_rag):
    prompt_juez = f"""
    Eres un JUEZ DE EXAMEN RIGUROSO. No regales puntos, pero valora el conocimiento técnico.

    PREGUNTA: {pregunta}
    RESPUESTA MAESTRA: {respuesta_esperada}
    RESPUESTA DEL ALUMNO (RAG): {respuesta_rag}

    CRITERIOS DE SUSPENSO (NO):
    1. Si el alumno dice "no se menciona", "no hay información" o "no hay datos", y la RESPUESTA MAESTRA sí los tiene -> NO.
    2. Si el alumno da una cifra económica diferente a la de la respuesta maestra -> NO.
    3. Si el alumno dice que algo es "No grave" y el profesor dice que es "Muy grave" (o viceversa) -> NO.

    CRITERIOS DE APROBADO (SÍ):
    1. Si el alumno explica el concepto legal correctamente aunque use sinónimos.
    2. Si el alumno incluye más detalles técnicos que el profesor, pero la base es correcta.

    Responde ÚNICAMENTE con 'SÍ' o 'NO'.
    """
    
    try:
        response = llm_judge.complete(prompt_juez)
        resultado = response.text.strip().upper()
        return "SÍ" in resultado
    except Exception as e:
        print(f" Error evaluando: {e}")
        return False

# --- PROCESO DE VALIDACIÓN ---

print(f"Iniciando evaluación inteligente con {JUDGE_NAME}...\n")

master_data = []
# Leemos el archivo maestro (el que tiene las respuestas correctas)
with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        master_data.append({
            "question": row["Question"].strip(),
            "expected": row["Answer"].strip()
        })

points = 0
total = 0

# Leemos los resultados del RAG
with open(OUTPUT_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for i, row in enumerate(reader):
        pregunta = master_data[i]["question"]
        esperada = master_data[i]["expected"]
        rag = row["rag_answer"]
        
        print(f"Evaluando Pregunta {i+1}...")
        es_valida = evaluar_con_ia(pregunta, esperada, rag)
        
        total += 1
        if es_valida:
            points += 1
            print(f"  > Resultado: ✅ CORRECTA")
        else:
            print(f"  > Resultado: ❌ INCORRECTA")
            print(f"    - Esperada: {esperada[:60]}...")
            print(f"    - Obtenida: {rag[:60]}...")

print("\n" + "#"*40)
print(f"  PUNTUACIÓN FINAL SEMÁNTICA: {points} / {total}")
print("#"*40)