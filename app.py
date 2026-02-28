import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_KEY"),
)

completion = client.chat.completions.create(
    model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
    messages=[
        {"role": "user", "content": "Explain RAG simply."}
    ],
    max_tokens=150,
    temperature=0.7,
)

print(completion.choices[0].message.content)