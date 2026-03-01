import os
import re
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
#masking function
def mask_text(text):
    #masking email
    text =re.sub(r'\S+@\S+','[EMAIL_MASKED]',text)

    #masking phone number
    text=re.sub(r'\b\d{10}\b','[PHONE_MASKED]',text)
    return text 


# Retrieval function

def retrieve(query, k=2):
    query_vector = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# RAG Function

def rag_answer(query):
    #masking user query
    masked_query=mask_text(query)
    #logging origial query
    print("\n[LOG] original query:")
    print(query)

    #logging masked query
    print("\n[LOG] masked query:")
    print(masked_query)

    #retrieving using masked query
    context_chunks = retrieve(masked_query)
    context = "\n".join(context_chunks) 

    #masking retrieved context
    context=mask_text(context)

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
            "content": f"Context:\n{context}\n\nQuestion:\n{masked_query}"
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



# CLI chat
if __name__ == "__main__":

    print("\n CLI RAG chat \n")

    while True:
        user_input = input("query: ")

        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting CLI chat...\n")
            break

        response = rag_answer(user_input)

        print("\LLM:", response)
        print("\n" + "-"*50 + "\n")