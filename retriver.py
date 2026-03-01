import numpy as np
from embedding import embed_text

def retrieve(query, vector_store, chunks, k=2):
    query_vector = embed_text([query])
    query_vector = np.array(query_vector).astype("float32")
    indices = vector_store.search(query_vector, k)
    return [chunks[i] for i in indices]