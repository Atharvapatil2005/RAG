import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# Load environment

load_dotenv()


# Load embedding model

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Load document

with open("data.txt", "r") as f:
    text = f.read()

chunks = [line for line in text.split("\n") if line.strip() != ""]


# Create embeddings

embeddings = embedder.encode(chunks)
embeddings = np.array(embeddings).astype("float32")


# Create FAISS index

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# Setup Zephyr (HF Router)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_API_KEY"]
)


# Retrieval function

def retrieve(query, k=2):
    query_vector = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# RAG Function

def rag_answer(query):
    #  Retrieve context
    context_chunks = retrieve(query)
    context = "\n".join(context_chunks) 

    print("\nRetrieved Context:\n")
    print(context)
    print("\n---\n")

    # Build prompt
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the provided context. If the answer is not in the context, say you don't know."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    #Call LLM
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )

    return completion.choices[0].message.content



# Test Query

question = "how do llm calls work?"

answer = rag_answer(question)

print("Final Answer:\n")
print(answer)
