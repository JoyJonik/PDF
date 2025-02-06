import os
import openai
import faiss
import json
import csv
import numpy as np
import PyPDF2  
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

openai.api_key = "Enter API key"

app = FastAPI()

EMBEDDING_DIM = 1536  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")

doc1_path = os.path.join(DOCS_DIR, "doc1.pdf")
doc2_path = os.path.join(DOCS_DIR, "doc2.pdf")
cards_path = os.path.join(DOCS_DIR, "cards.csv")

def read_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def read_csv_text(csv_path):
    lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            line = " | ".join(row)
            lines.append(line)
    return "\n".join(lines)

text_doc1 = read_pdf_text(doc1_path)
text_doc2 = read_pdf_text(doc2_path)
text_csv  = read_csv_text(cards_path)

def chunk_text(text, chunk_size=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end
    return chunks

all_texts = []

chunks_doc1 = chunk_text(text_doc1, chunk_size=500)
for i, ch in enumerate(chunks_doc1):
    all_texts.append({
        "id": f"doc1_{i}",
        "text": ch,
        "metadata": {"source": "doc1.pdf"}
    })

chunks_doc2 = chunk_text(text_doc2, chunk_size=500)
for i, ch in enumerate(chunks_doc2):
    all_texts.append({
        "id": f"doc2_{i}",
        "text": ch,
        "metadata": {"source": "doc2.pdf"}
    })

chunks_csv = chunk_text(text_csv, chunk_size=500)
for i, ch in enumerate(chunks_csv):
    all_texts.append({
        "id": f"cards_{i}",
        "text": ch,
        "metadata": {"source": "cards.csv"}
    })

print("Генерация эмбеддингов для всех фрагментов...")

vectors = []  
documents_meta = []  

for doc in all_texts:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=doc["text"]
    )
    emb = response["data"][0]["embedding"]
    vectors.append(emb)
    documents_meta.append(doc) 

vectors_np = np.array(vectors, dtype="float32")

index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(vectors_np)

print(f"Загружено {len(documents_meta)} фрагментов в Faiss индекс.")


class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: QueryRequest):
    question = payload.question.strip()

    q_emb_response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=question
    )
    q_emb = np.array(q_emb_response["data"][0]["embedding"], dtype="float32").reshape(1, EMBEDDING_DIM)

    k = 3
    distances, ids = index.search(q_emb, k)

    relevant_chunks = []
    for idx in ids[0]:
        chunk_data = documents_meta[idx]
        relevant_chunks.append(chunk_data["text"])

    system_prompt = (
        "Ты – интеллектуальный помощник. У тебя есть набор фрагментов из документов (pdf и csv). "
        "Используй фрагменты, чтобы ответить на вопрос. Если ответа нет в этих фрагментах, "
        "ответь максимально правдиво и кратко. Если нужно, можешь формулировать ответ от себя."
    )

    context_text = "\n\n".join(relevant_chunks)
    max_context_len = 3000  
    if len(context_text) > max_context_len:
        context_text = context_text[:max_context_len]

    full_prompt = f"{system_prompt}\n\nФРАГМЕНТЫ:\n{context_text}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}\n\nОТВЕТ:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
        max_tokens=300
    )
    answer = response["choices"][0]["message"]["content"]

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
