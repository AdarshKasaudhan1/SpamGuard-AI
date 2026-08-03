"""
app.py
------
Flask application for the Professional AI Spam Guard.

Routes:
  GET  /                → Serves the dashboard UI
  POST /predict-text    → Classifies raw text input
  POST /predict-image   → Extracts text via OCR then classifies
"""

import os
import re
import io
import logging
import tempfile

import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Load persisted ML artefacts ───────────────────────────────────────────────
MODEL_PATH      = os.path.join("model", "spam_model.joblib")
VECTORIZER_PATH = os.path.join("model", "vectorizer.joblib")

try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("✅  ML model and vectorizer loaded successfully.")
except FileNotFoundError:
    logger.error(
        "❌  Model files not found. Please run `python train_model.py` first."
    )
    model = vectorizer = None

# Allowed image extensions for the OCR route
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}


# ── Helper functions ───────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Return True if the filename has a permitted image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    text = text.lower()

    # Replace URLs with token
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # Replace phone numbers
    text = re.sub(r"\b\d{10,}\b", " PHONE ", text)

    # Replace numbers
    text = re.sub(r"\d+", " NUMBER ", text)

    # Keep useful symbols (₹ $ % !)
    text = re.sub(r"[^a-z0-9₹$%!\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classify(text: str) -> dict:
    """
    Run the spam classifier on pre-cleaned text.

    Returns a dict with:
      - label       : "spam" | "ham"
      - confidence  : float 0-100 (probability of the predicted class)
      - spam_prob   : float 0-100 (raw spam probability – drives the meter)
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model not loaded. Run train_model.py first.")

    cleaned   = clean_text(text)
    features  = vectorizer.transform([cleaned])

    # predict_proba returns [P(ham), P(spam)]
    proba     = model.predict_proba(features)[0]
    spam_prob = float(proba[1]) * 100
    ham_prob  = float(proba[0]) * 100

    label      = "spam" if spam_prob > 50 else "ham"
    confidence = spam_prob if label == "spam" else ham_prob

    return {
        "label":      label,
        "confidence": round(confidence, 2),
        "spam_prob":  round(spam_prob, 2),
        "ham_prob":   round(ham_prob, 2),
    }


def extract_text_easyocr(image_bytes: bytes) -> str:
    """
    Use EasyOCR to extract text from raw image bytes.
    EasyOCR is GPU-optional and handles a wide variety of fonts well.
    """
    try:
        import easyocr
        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        # Initialise reader (downloads model weights on first run)
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(img_array, detail=0)
        return " ".join(results)
    except ImportError:
        raise RuntimeError("easyocr not installed. Run: pip install easyocr")


def extract_text_tesseract(image_bytes: bytes) -> str:
    """
    Fallback OCR using pytesseract (requires Tesseract binary on PATH).
    """
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except ImportError:
        raise RuntimeError(
            "pytesseract not installed. Run: pip install pytesseract\n"
            "Also install the Tesseract binary: https://github.com/tesseract-ocr/tesseract"
        )


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Try EasyOCR first; fall back to Tesseract if EasyOCR is unavailable.
    """
    try:
        return extract_text_easyocr(image_bytes)
    except RuntimeError:
        logger.warning("EasyOCR unavailable – falling back to Tesseract.")
        return extract_text_tesseract(image_bytes)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/predict-text", methods=["POST"])
def predict_text():
    """
    Classify a plain-text message.

    Request body (JSON):  { "text": "You have won a free prize!" }
    Response (JSON):      { label, confidence, spam_prob, ham_prob, extracted_text }
    """
    try:
        payload = request.get_json(force=True)
        if not payload or "text" not in payload:
            return jsonify({"error": "Missing 'text' field in request body."}), 400

        text = payload["text"].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty."}), 400

        result = classify(text)
        result["extracted_text"] = text
        logger.info("Text prediction → %s (%.1f%%)", result["label"], result["spam_prob"])
        return jsonify(result)

    except RuntimeError as exc:
        logger.error("Prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Unexpected error in /predict-text")
        return jsonify({"error": "Internal server error."}), 500


@app.route("/predict-image", methods=["POST"])
def predict_image():
    """
    Accept an uploaded image, run OCR to extract text, then classify.

    Multipart form field: 'image'
    Response (JSON): { label, confidence, spam_prob, ham_prob, extracted_text }
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 415

        image_bytes = file.read()

        # ── OCR ───────────────────────────────────────────────────────────────
        extracted_text = extract_text_from_image(image_bytes)
        if not extracted_text.strip():
            return jsonify({"error": "Could not extract any text from the image."}), 422

        logger.info("OCR extracted: %.80s …", extracted_text.replace("\n", " "))

        # ── Classification ────────────────────────────────────────────────────
        result = classify(extracted_text)
        result["extracted_text"] = extracted_text.strip()
        logger.info("Image prediction → %s (%.1f%%)", result["label"], result["spam_prob"])
        return jsonify(result)

    except RuntimeError as exc:
        logger.error("OCR/model error: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Unexpected error in /predict-image")
        return jsonify({"error": "Internal server error."}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
