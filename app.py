
import os
import uuid
import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify

# -----------------------------
# Animal Info Loader (official)
# -----------------------------
from animal_info_loader import get_animal_info

# -----------------------------
# Prediction functions
# -----------------------------
from predict_level1 import predict_level1_image_yolo as predict_level1_image
from predict_level2 import predict_image as predict_level2_image
from predict_level3 import predict_image as predict_level3_image

# -----------------------------
# Flask setup
# -----------------------------
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------
# File Formats
# -----------------------------
ALLOWED_EXT = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# -----------------------------
# Safe Info (prevents crashes)
# -----------------------------
DEFAULT_INFO = {
    "scientific_name": "undefined",
    "habitat": "undefined",
    "life_span": "undefined",
    "diet": "undefined",
    "other_info": "undefined",
}

def safe_merge_info(pred):
    """Return guaranteed info dict (even in fallback / unknown cases)."""
    if not pred:
        return DEFAULT_INFO.copy()

    data = get_animal_info(pred)
    final = DEFAULT_INFO.copy()

    if isinstance(data, dict):
        final.update(data)

    return final


# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
#                    LEVEL 1 — UPLOAD
# ============================================================
@app.route("/level1", methods=["GET", "POST"])
def level1():
    result = None
    uploaded_file = None
    cropped_file = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Invalid image format. Allowed: JPG, JPEG, PNG"
        else:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)
            uploaded_file = filename

            result = predict_level1_image(filepath)

            # Cropped result (if exists)
            cropped_path = result.get("cropped_path")
            if cropped_path:
                cropped_file = os.path.basename(cropped_path)

    return render_template("level1.html",
                           uploaded_file=uploaded_file,
                           cropped_file=cropped_file,
                           result=result,
                           error=error)


# ============================================================
#                    LEVEL 1 — CAMERA
# ============================================================
@app.route("/capture", methods=["POST"])
def capture():
    try:
        data = request.json.get("image")
        if not data:
            return jsonify({"status": "error", "message": "No image data"})

        img_bytes = base64.b64decode(data.split(",")[1])
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        result = predict_level1_image(filepath)

        # Fix cropped path for frontend
        if result.get("cropped_path"):
            result["cropped_path"] = "uploads/" + os.path.basename(result["cropped_path"])
            result.pop("segmented_path", None)

        return jsonify({"status": "success", "filename": filename, "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



# ============================================================
#                    LEVEL 2 — UPLOAD
# ============================================================

# ============================================================
#                    LEVEL 2 — UPLOAD
# ============================================================
@app.route("/level2", methods=["GET", "POST"])
def level2():
    result = None
    uploaded_file = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Invalid image format. Allowed: JPG, JPEG, PNG"
        else:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)
            uploaded_file = filename

            result = predict_level2_image(filepath)

            # ----------------------------
            # RAG fallback → do NOT show info
            # ----------------------------
            if result.get("rag_fallback"):
                result["show_info"] = False
                return render_template(
                    "level2.html",
                    uploaded_file=uploaded_file,
                    result=result,
                    error=error
                )

            # ----------------------------
            # Normal CNN prediction → show animal info
            # ----------------------------
            if "top_classes" in result and "top_probs" in result:
                result["top"] = list(zip(result["top_classes"], result["top_probs"]))

            pred = result.get("prediction", "").lower()

            # merge animal info into result so template can access top-level keys
            info = safe_merge_info(pred)
            result.update(info)
            result["show_info"] = True

    return render_template(
        "level2.html",
        uploaded_file=uploaded_file,
        result=result,
        error=error
    )


# ============================================================
#                    LEVEL 2 — CAMERA
# ============================================================
@app.route("/capture_level2", methods=["POST"])
def capture_level2():
    try:
        data = request.json.get("image")
        if not data:
            return jsonify({"status": "error", "message": "No image data"})

        img_bytes = base64.b64decode(data.split(",")[1])
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        result = predict_level2_image(filepath)

        # ----------------------------
        # RAG fallback → do NOT show info
        # ----------------------------
        if result.get("rag_fallback"):
            result["show_info"] = False
            return jsonify({"status": "success", "filename": filename, "result": result})

        # ----------------------------
        # Normal CNN prediction → show animal info
        # ----------------------------
        if "top_classes" in result and "top_probs" in result:
            result["top"] = list(zip(result["top_classes"], result["top_probs"]))

        pred = result.get("prediction", "").lower()
        info = safe_merge_info(pred)
        result.update(info)
        result["show_info"] = True

        return jsonify({"status": "success", "filename": filename, "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ============================================================
#                    LEVEL 3 — UPLOAD
# ============================================================
@app.route("/level3", methods=["GET", "POST"])
def level3():
    result = None
    uploaded_file = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = "Invalid image format. Allowed: JPG, JPEG, PNG"
        else:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)
            uploaded_file = filename

            result = predict_level3_image(filepath)

            if result.get("unknown"):
                result["top"] = []

            elif "closest_species" in result:
                result["top"] = [
                    (sp["species"], sp["distance"]) 
                    for sp in result["closest_species"]
                ]

    return render_template("level3.html",
                           uploaded_file=uploaded_file,
                           result=result,
                           error=error)



# -----------------------------
# Static route for uploaded files
# -----------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)



# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
