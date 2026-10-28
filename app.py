"""
app.py
------
Flask application for the Professional AI Spam Guard.

Routes:
  GET  /                → Serves the dashboard UI
  POST /predict-text    → Classifies raw text input
  POST /predict-image   → Extracts text via OCR then classifies
"""

import io
import logging
import os
import sys
from typing import Optional

import joblib
from flask import Flask, jsonify, render_template, request

from preprocess import clean_text

# Reconfigure stdout/stderr to UTF-8 for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Load persisted ML artefacts ───────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "spam_model.joblib")
VECTORIZER_PATH = os.path.join("model", "vectorizer.joblib")

model = None
vectorizer = None


def load_model_artifacts():
    """Load or reload model and vectorizer artifacts from disk."""
    global model, vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("✅  ML model and vectorizer loaded successfully.")
        return True
    except FileNotFoundError:
        logger.error(
            "❌  Model files not found. Please run `python train_model.py` first."
        )
        model = vectorizer = None
        return False


# Initial load at startup
load_model_artifacts()

# Allowed image extensions for the OCR route
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}

# Global singleton EasyOCR reader cache
_easyocr_reader = None


def get_easyocr_reader():
    """
    Lazy-initialize and return the cached singleton EasyOCR Reader.
    Avoids re-instantiating PyTorch models and reloading weights on every image request.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr

            logger.info("Initializing EasyOCR reader (cached singleton)...")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("✅ EasyOCR reader initialized and cached.")
        except ImportError:
            raise RuntimeError("easyocr not installed. Run: pip install easyocr")
        except Exception as exc:
            logger.error("Failed to initialize EasyOCR reader: %s", exc)
            raise RuntimeError(f"EasyOCR initialization failed: {exc}")
    return _easyocr_reader


# ── Helper functions ───────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Return True if the filename has a permitted image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify(text: str) -> dict:
    """
    Run the spam classifier on text using the shared clean_text pipeline.

    Returns a dict with:
      - label       : "spam" | "ham"
      - confidence  : float 0-100 (probability of the predicted class)
      - spam_prob   : float 0-100 (raw spam probability – drives the meter)
      - ham_prob    : float 0-100 (raw ham probability)
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        if not load_model_artifacts():
            raise RuntimeError("Model not loaded. Run train_model.py first.")

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    # predict_proba returns [P(ham), P(spam)]
    proba = model.predict_proba(features)[0]
    spam_prob = float(proba[1]) * 100
    ham_prob = float(proba[0]) * 100

    label = "spam" if spam_prob > 50 else "ham"
    confidence = spam_prob if label == "spam" else ham_prob

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "spam_prob": round(spam_prob, 2),
        "ham_prob": round(ham_prob, 2),
    }


def extract_text_easyocr(image_bytes: bytes) -> str:
    """
    Use cached EasyOCR singleton to extract text from raw image bytes.
    """
    reader = get_easyocr_reader()
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    results = reader.readtext(img_array, detail=0)
    return " ".join(results)


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
    Try EasyOCR first; fall back to Tesseract if EasyOCR is unavailable or encounters an error.
    """
    try:
        return extract_text_easyocr(image_bytes)
    except Exception as exc:
        logger.warning(
            "EasyOCR extraction failed (%s) – falling back to Tesseract.", exc
        )
        try:
            return extract_text_tesseract(image_bytes)
        except Exception as tess_exc:
            logger.error("Tesseract fallback also failed: %s", tess_exc)
            raise RuntimeError(
                f"Image OCR failed on both EasyOCR and Tesseract: {exc}"
            )


# ── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle files exceeding the 10 MB limit."""
    return jsonify({"error": "File size exceeds the 10 MB limit."}), 413


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
        payload = request.get_json(silent=True)
        if not payload or not isinstance(payload, dict) or "text" not in payload:
            return jsonify({"error": "Missing 'text' field in request body."}), 400

        text = payload.get("text")
        if text is None or not str(text).strip():
            return jsonify({"error": "Text cannot be empty."}), 400

        text = str(text).strip()
        result = classify(text)
        result["extracted_text"] = text
        logger.info(
            "Text prediction → %s (%.1f%% spam prob)",
            result["label"],
            result["spam_prob"],
        )
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
            return jsonify(
                {
                    "error": (
                        f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    )
                }
            ), 415

        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # ── OCR ───────────────────────────────────────────────────────────────
        extracted_text = extract_text_from_image(image_bytes)
        if not extracted_text or not extracted_text.strip():
            return (
                jsonify({"error": "Could not extract any text from the image."}),
                422,
            )

        logger.info("OCR extracted: %.80s …", extracted_text.replace("\n", " "))

        # ── Classification ────────────────────────────────────────────────────
        result = classify(extracted_text)
        result["extracted_text"] = extracted_text.strip()
        logger.info(
            "Image prediction → %s (%.1f%% spam prob)",
            result["label"],
            result["spam_prob"],
        )
        return jsonify(result)

    except RuntimeError as exc:
        logger.error("OCR/model error: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Unexpected error in /predict-image")
        return jsonify({"error": "Internal server error."}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=debug, host=host, port=port)
