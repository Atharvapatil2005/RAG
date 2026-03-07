from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import generator
from rag_pipeline import rag_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    #rag returns generator 
    generator =rag_answer(query.question)
    return StreamingResponse(generator,media_type="text/plain")
