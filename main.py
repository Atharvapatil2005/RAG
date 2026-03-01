from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import rag_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = rag_answer(query.question)
    return {"answer": answer}