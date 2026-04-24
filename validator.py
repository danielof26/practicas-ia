import os
import csv

MODEL_NAME = "qwen3.5:4b"

DOCS_FOLDER = "/Users/danielonisfabian/Desktop/GRUPO_INVESTIGACION/docs"

QUESTIONS_CSV = os.path.join(DOCS_FOLDER, "source_doc/questions_f1.csv")
OUTPUT_CSV = os.path.join(DOCS_FOLDER, f"results/rag_answers_{MODEL_NAME}.csv")


def print_spaces():
    print()
    print()
    

keywords = []

with open(QUESTIONS_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        keywords.append(row["Keywords"].strip())
        
points = 0

with open(OUTPUT_CSV, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for i, row in enumerate(reader):
        answer = row["rag_answer"].lower()
        keys = [k.strip().lower() for k in keywords[i].split(',')]
        
        all_found = True
        correct_answers = 0

        for k in keys:
            if k not in answer:
                all_found = False
            else:
                correct_answers += 1

        
        
        print(f"Question {i+1} answered {(correct_answers/len(keys))*100:.2f}% correctly!!")
        
        if(all_found):
            points += 1

    
        
print_spaces()

print(f"############################### FINAL POINTS: {points} ###############################")

print_spaces()