import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY").strip()

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://example.com",
    "X-OpenRouter-Title": "RAGProject",
    "Content-Type": "application/json",
}

data = {
    "model": "upstage/solar-pro-3:free",
    "messages": [
        {"role": "user", "content": "Explain RAG simply."}
    ],
    "provider": {
        "allow_fallbacks": True
    }
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=data,
)

print("STATUS:", response.status_code)
print("RESPONSE:", response.text)