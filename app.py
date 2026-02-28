from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("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"),   # explicitly pass
)

response = model.invoke("Explain RAG in simple terms.")
print(response.content)