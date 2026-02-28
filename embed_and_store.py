import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Loaded a neural network embedding model to convert text -> vectors
#It maps meaning into 384-dimensional space.
#Sentences with similar meaning end up closer together in that space.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load data from file 
with open("data.txt", "r") as f:
    text = f.read()

#chunking the document into individual lines
#because it allows fine-grained retrieval 
chunks = [line for line in text.split("\n") if line.strip() != ""]

# Converting chunks →  vector embeddings
#float 32 format because FAISS accepts it
embeddings = embedder.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
# this stores vectors and computes similarity, returns nearest neighbours
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Stored", len(chunks), "chunks in FAISS.")
print("Vector dimension:", dimension)
