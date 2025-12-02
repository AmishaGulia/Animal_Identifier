


# Animal Identifier

A multi-level animal detection and classification system that leverages **YOLO**, **CNNs**, **Vision Transformers (ViT)**, **FAISS**, and **Large Language Models (LLM)** to detect, classify, and explain animal species in images or live camera feed. The system supports community contributions for unknown species.

---

## 🔹 Features

- **Level 1 – Kids / Basic Detection**
  - Real-time detection using YOLOv8.
  - Classification with ResNet18.
  - Simple and fast predictions for common species.

- **Level 2 – Professionals / Rare Species**
  - Classification using ResNet50.
  - RAG (Retrieval-Augmented Generation) fallback for uncertain predictions.
  - Displays detailed animal info (scientific name, habitat, diet, lifespan, etc.).

- **Level 3 – Community / Open Database**
  - Embedding-based retrieval using Vision Transformer (ViT) + FAISS.
  - Detects unknown species using distance threshold.
  - Generates explanations using LLM.
  - Allows users to add new species to the database.

---

## 🗂 Project Structure

project/
├── app.py # Main Flask application
├── templates/ # HTML templates (Level1, Level2, Level3)
├── static/uploads/ # Uploaded images
├── predict_level1.py # YOLO + ResNet18 pipeline
├── predict_level2.py # ResNet50 + RAG pipeline
├── predict_level3.py # ViT embeddings + FAISS + LLM pipeline
├── rag_service.py # RAG service with FAISS and LLM
├── animal_info_loader.py # Loads animal information JSON
├── embeddings.npy # Precomputed embeddings for FAISS
├── labels.npy # Labels for FAISS index
├── faiss_index.bin # FAISS index file
└── README.md # This file


---

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/animal-classifier.git
cd animal-classifier

    Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

    Install dependencies:

pip install -r requirements.txt

    Download YOLOv8 weights if not included:

ultralytics yolov8s.pt

    Place dataset images in structured folders (for Level 3 embeddings):

dataset/
├── lion/
├── tiger/
├── elephant/
...

🚀 Running the Application

Start the Flask app:

python app.py

Access in browser: http://localhost:5000

    Level 1: /level1

    Level 2: /level2

    Level 3: /level3

🔧 How it Works

    Level 1:

        Upload or capture image.

        YOLO detects animal bounding box.

        ResNet18 predicts species.

        Returns top prediction with confidence.

    Level 2:

        ResNet50 predicts species.

        If confidence < threshold → RAG fallback.

        Displays top predictions and animal info.

    Level 3:

        ViT generates embeddings.

        FAISS retrieves nearest species.

        Detects unknown species based on distance.

        Generates LLM explanation.

        Users can add new species to database.

⚙️ Add a New Species (Level 3)

    Upload an unknown species image.

    Click Add Species to Database.

    Enter species name.

    Image embedding is added to FAISS index for future predictions.

📚 Dependencies

    Python 3.10+

    PyTorch

    torchvision

    timm

    FAISS

    Pillow

    OpenCV

    Ultralytics YOLO

    Transformers (HuggingFace)

    Flask

    Numpy

🧠 Future Work

    Expand community species database.

    Multi-animal detection in real-time video.

    LLM explanations enhanced with habitat and behavior info.

    Mobile and API integration.

📄 References

    YOLOv8 – Ultralytics YOLO

ResNet / CNN – Torchvision Models

Vision Transformers – TIMM

FAISS – FAISS: Facebook AI Similarity Search

HuggingFace Transformers – T5 / FLAN-T5

