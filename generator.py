import os
from openai import OpenAI
from dotenv import load_dotenv


# Load environment

load_dotenv()
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_API_KEY"]
)
#llm build prompt
def generate_answer(context, query):
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using provided context. If not found, say you don't know."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]
#completion model used
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )

    return completion.choices[0].message.content