from sentence_transformers import SentenceTransformer

# Load embedding model

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    return embedder.encode(texts)