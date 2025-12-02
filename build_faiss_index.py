import faiss
import numpy as np

# Load embeddings + labels
emb = np.load("embeddings.npy")
labels = np.load("labels.npy")

# Embedding dimension
dim = emb.shape[1]

# Create FAISS L2 index
index = faiss.IndexFlatL2(dim)

# Add all embeddings to index
index.add(emb)

# Save index + labels
faiss.write_index(index, "faiss_index.bin")
np.save("labels.npy", labels)

print("FAISS index created successfully.")
