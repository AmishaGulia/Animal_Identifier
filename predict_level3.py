

import os
import numpy as np
from PIL import Image
from rag_service import RAGServiceLLM as RAGService


FAISS_INDEX = "faiss_index.bin"
LABELS_FILE = "labels.npy"
TOP_K = 5
UNKNOWN_THRESHOLD = 0.75   

# Initialize RAG Service

rag = RAGService()


# Prediction / Retrieval function

def predict_image(img_path, top_k=TOP_K, threshold=UNKNOWN_THRESHOLD):
    """
    Level 3: Community / Open Database
    - Retrieves nearest neighbors from FAISS
    - Detects unknown species using distance threshold
    - Generates explanation using LLM
    """
    img = Image.open(img_path).convert("RGB")

    # Step 1: Retrieve nearest neighbors
    neighbors = rag.retrieve(img, top_k=top_k)   # [(species, distance), ...]

    # Extract distances
    distances = [dist for (_, dist) in neighbors]

    # Step 2: Unknown species detection
    is_unknown = all(dist > threshold for dist in distances)

    if is_unknown:
        return {
            "uploaded_image": os.path.basename(img_path),
            "predicted_species": "UNKNOWN_SPECIES",
            "closest_species": [
                {"species": sp, "distance": dist} for (sp, dist) in neighbors
            ],
            "is_unknown": True,
            "explanation": (
                "This image does not closely match any species in the FAISS database. "
                "It might be a new or unseen species. You may add it to the database "
                "using the add_to_database() function."
            )
        }

    # Step 3: Known species case
    top_species = [sp for (sp, _) in neighbors]

    # Generate contextual explanation using RAG/LLM
    explanation = rag.generate_explanation(top_species)

    return {
        "uploaded_image": os.path.basename(img_path),
        "predicted_species": neighbors[0][0],   # best match
        "closest_species": [
            {"species": sp, "distance": dist} for (sp, dist) in neighbors
        ],
        "is_unknown": False,
        "explanation": explanation
    }


# Add new image to database (optional)

def add_to_database(img_path, label, embeddings_file="embeddings.npy"):
    """
    Adds a new image embedding and label to the FAISS index.
    """
    img = Image.open(img_path).convert("RGB")
    emb = rag.embed(img)  # generate embedding

    # Load existing embeddings
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
        embeddings = np.vstack([embeddings, emb])
    else:
        embeddings = emb

    np.save(embeddings_file, embeddings)

    # Update FAISS index
    rag.index.add(emb)

    # Update labels
    labels = np.load(LABELS_FILE)
    labels = np.append(labels, label)
    np.save(LABELS_FILE, labels)

    return {"status": "added", "label": label}


if __name__ == "__main__":
    test_img = "/home/amisha/RAGANI/animal_dataset/alligator/Image_2.jpg"
    result = predict_image(test_img)
    print(result)
