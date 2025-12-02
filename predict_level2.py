
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
import cv2
import numpy as np
from ultralytics import YOLO
import json

from rag_service import RAGServiceLLM as RAGService
from animal_info_loader import get_animal_info


# LOAD ANIMAL INFO JSON

with open("animal_info.json", "r") as f:
    ANIMAL_INFO = json.load(f)

def get_animal_info(name):
    name = name.lower().strip()
    return ANIMAL_INFO.get(name, None)



CONF_THRESHOLD = 0.55
CLASS_FILE = "classes.txt"



# LOAD CLASS NAMES

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]
NUM_CLASSES = len(CLASS_NAMES)



# LOAD CNN MODEL (ResNet50)

cnn_model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
cnn_model.load_state_dict(
    torch.load("/home/amisha/RAGANI/resnet50_split_trainval.pth", map_location="cpu")
)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



yolo_model = YOLO("yolov8s.pt")



# LOAD RAG

rag = RAGService()


# CLASSIFY CROPPED REGION

def classify_crop(crop_bgr):

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tensor = cnn_transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = cnn_model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, k=3)
    top_probs = top_probs.tolist()
    top_idx = top_idx.tolist()

    top_classes = [CLASS_NAMES[i] for i in top_idx]

    conf = float(top_probs[0])
    pred = top_classes[0]

    # ---------------- Fallback ----------------
    if conf < CONF_THRESHOLD:
        species = rag.retrieve(img)
        explanation = rag.generate_explanation([sp[0] for sp in species])

        return {
            "prediction": "fallback",
            "confidence": 0.0,                     
            "closest_species": species,
            "explanation": explanation,
            "top_classes": top_classes,
            "top_probs": top_probs
        }

    return {
        "prediction": pred,
        "confidence": float(conf),                  # always float
        "top_classes": top_classes,
        "top_probs": top_probs
    }



# YOLO DETECT + CLASSIFY

def detect_and_classify(frame):

    results = yolo_model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det_conf = float(box.conf[0])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_result = classify_crop(crop)

        label = (
            cls_result["prediction"]
            if cls_result["prediction"] != "fallback"
            else cls_result["closest_species"][0][0]
        )

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Label text
        cv2.putText(frame,
                    f"{label} ({det_conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "label": label,
            "det_conf": det_conf,
            "classification": cls_result
        })

    return frame, detections



# FLASK IMAGE PREDICTION

def predict_image(image_path):
    """Used by Flask Level 2 upload system."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not read image")

    frame, detections = detect_and_classify(frame)

    # No detection
    if len(detections) == 0:
        return {
            "prediction": "No animal detected",
            "confidence": 0.0,                       # <<< FIXED
            "top_classes": [],
            "top_probs": [],
            "info": None
        }

    det = detections[0]
    cls = det["classification"]

    # ----------------- Fallback (RAG) -----------------
    if cls["prediction"] == "fallback":
        fallback_name = cls["closest_species"][0][0]
        info = get_animal_info(fallback_name)

        return {
            "prediction": "fallback",
            "confidence": 0.0,                       
            "top_classes": cls["top_classes"],
            "top_probs": cls["top_probs"],
            "closest_species": cls["closest_species"],
            "explanation": cls["explanation"],
            "info": info
        }

    # ----------------- Normal CNN -----------------
    animal = cls["prediction"]
    info = get_animal_info(animal)

    return {
        "prediction": animal,
        "confidence": cls["confidence"],             
        "top_classes": cls["top_classes"],
        "top_probs": cls["top_probs"],
        "info": info
    }



# WEBCAM MODE

def run_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not access camera")
        return

    print("\n🎥 Live YOLO + CNN + RAG Mode (Press Q to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detections = detect_and_classify(frame)

        cv2.imshow("Animal Detector + Classifier", frame)

        if cvv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_webcam()
