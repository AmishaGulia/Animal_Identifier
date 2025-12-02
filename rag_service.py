
import faiss
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RAGServiceLLM:
    def __init__(self):
      
        # Load FAISS index and labels
        
        self.index = faiss.read_index("faiss_index.bin")
        self.labels = np.load("labels.npy")

        
        # Vision Transformer embedder
        
        self.embedder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0  # return embeddings
        )
        self.embedder.eval()

       
        # Large LLM for explanations
       
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.generator.eval()

        
        # Image transform pipeline
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    
    # Generate embeddings from image
  
    def embed(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            emb = self.embedder(img_tensor)
        return emb.cpu().numpy().astype("float32")  # FAISS requires float32

    
    # Retrieve closest species via FAISS
    
    def retrieve(self, image: Image.Image, top_k: int = 3):
        emb = self.embed(image)
        distances, idx = self.index.search(emb, top_k)

        results = [(self.labels[i], float(distances[0][rank])) for rank, i in enumerate(idx[0])]
        return results

    
    # Generate LLM explanation
    
    def generate_explanation(self, species_list):
        prompt = (
            "Fallback system activated. "
            "The closest species retrieved are: "
            + ", ".join(species_list) +
            ". Explain the key visual differences between these species in simple terms."
        )

        tokens = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.generator.generate(**tokens, max_length=150)
        explanation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return explanation
