
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from animal_info_loader import get_animal_info




CONF_THRESHOLD = 0.55
CLASS_FILE = "/home/amisha/RAGANI/classes_level1.txt"
RESNET18_MODEL_PATH = "/home/amisha/RAGANI/best_resnet18.pth"



# LOAD CLASS LABELS

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]
NUM_CLASSES = len(CLASS_NAMES)



# LOAD RESNET-18 MODEL

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state = torch.load(RESNET18_MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])



# DIRECT IMAGE PREDICTION (UPLOAD)

def predict_level1_image(image_path):

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)
    top_prob = float(top_prob)
    pred_class = CLASS_NAMES[int(top_idx)]

    # RUN JSON LOOKUP ⭐
    info = get_animal_info(pred_class)

    if top_prob < CONF_THRESHOLD:
        return {
            "prediction": "Not in dataset",
            "confidence": top_prob,
            "top_classes": CLASS_NAMES,
            "top_probs": probs.tolist(),
            "cropped_path": None,
            "info": None     
        }

    return {
        "prediction": pred_class,
        "confidence": top_prob,
        "top_classes": CLASS_NAMES,
        "top_probs": probs.tolist(),
        "cropped_path": None,
        "info": info      
    }



# YOLO MODEL FOR CROPPING
yolo = YOLO("yolov8s.pt")



# CLASSIFY CROPPED YOLO IMAGE

def classify_crop(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)
    top_prob = float(top_prob)
    pred_class = CLASS_NAMES[int(top_idx)]

    if top_prob < CONF_THRESHOLD:
        return "Not in dataset", top_prob

    return pred_class, top_prob



# YOLO + RESNET-18 (WEBCAM OR CAMERA)

def predict_level1_image_yolo(image_path):

    frame = cv2.imread(image_path)
    if frame is None:
        return {
            "prediction": "Error: cannot read image",
            "confidence": None,
            "cropped_path": None,
            "info": None
        }

    results = yolo(frame, verbose=False)[0]

    if len(results.boxes) == 0:
        return {
            "prediction": "No animal detected",
            "confidence": None,
            "cropped_path": None,
            "info": None
        }

    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return {
            "prediction": "Crop error",
            "confidence": None,
            "cropped_path": None,
            "info": None
        }

    pred, conf = classify_crop(crop)

    cropped_path = image_path.replace(".png", "_crop.png")
    cv2.imwrite(cropped_path, crop)

   
    info = get_animal_info(pred)

    return {
        "prediction": pred,
        "confidence": conf,
        "cropped_path": cropped_path,
        "info": info     
    }
