import os
import csv
import spacy

MODEL_NAME = "qwen3:4b"

TEMA = "leyes"

DOCS_FOLDER = "/Users/danielonisfabian/Desktop/GRUPO_INVESTIGACION/docs"

QUESTIONS_CSV = os.path.join(DOCS_FOLDER, f"source_doc/{TEMA}/questions_{TEMA}.csv")
RAG_CSV = os.path.join(DOCS_FOLDER, f"results/{TEMA}/rag_answers_{MODEL_NAME}.csv")
OUTPUT_CSV = os.path.join(DOCS_FOLDER, f"results/{TEMA}/validation_{TEMA}_{MODEL_NAME}.csv")

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("es_core_news_sm")

def lematize(texto):
    doc = nlp(texto.lower())
    return [token.lemma_ for token in doc]


def validate(rag_answer, keywords_str):
    sem_answer = lematize(rag_answer)
    key_list = [k.strip() for k in keywords_str.split(',')]

    n_found = 0
    details = []

    for k in key_list:
        sem_k = lematize(k)
        found = any(s in sem_answer for s in sem_k)
        n_found += 1 if found else 0
        details.append(f"{k} = {'✅' if found else '❌'}")
    
    return (n_found / len(key_list)), ",".join(details)
        


keywords = []

with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        keywords.append(row["Keywords"].strip())


results = []

with open(RAG_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for i, row in enumerate(reader):
        question = row["question"]
        rag_answer = row["rag_answer"]
        keys = keywords[i]

        score, details = validate(rag_answer, keys)

        print(f"[{i+1}] Score: {score} | {details}")
        print(f"  Q: {question}")
        print(f"  RAG: {rag_answer}\n")

        results.append({
            "question":  question,
            "rag_answer": rag_answer,
            "keywords":  keys,
            "score":     score,
            "details":   details
        })

'''
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["question", "rag_answer", "keywords", "score", "details"], delimiter=';')
    writer.writeheader()
    writer.writerows(results)
'''