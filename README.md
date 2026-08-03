# 🛡️ SpamGuard AI — Multimodal Spam Detection System

A professional, portfolio-ready SaaS-style spam detection dashboard built with
**Flask + Scikit-learn + EasyOCR**.  Detects spam from both plain text and
uploaded images using OCR extraction.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Text classification** | TF-IDF bigrams + Multinomial Naive Bayes |
| **Image OCR scanning** | EasyOCR (GPU-optional) or Tesseract fallback |
| **Confidence meter** | Animated progress bar showing spam probability |
| **Scan history** | Last 5 results persisted in `localStorage` |
| **Dark dashboard UI** | Glassmorphism, animated grid, amber accent system |
| **Error handling** | Toast notifications + full try/except coverage |

---

## 📁 Project Structure

```
spam_guard/
├── app.py                  # Flask backend (API routes)
├── train_model.py          # One-time model training script
├── requirements.txt        # Python dependencies
├── model/                  # Auto-created after training
│   ├── spam_model.joblib
│   └── vectorizer.joblib
├── data/                   # Auto-created during training
│   └── SMSSpamCollection
├── templates/
│   └── index.html          # Dashboard UI
└── static/
    └── css/
        └── style.css       # Glassmorphism dark theme
```

---

## 🚀 Quick Start

### 1. Create & activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note — EasyOCR** will download its model weights (~100 MB) on the **first
> image scan**, not at install time.

#### Alternative: Tesseract (if you prefer not to use EasyOCR)

```bash
# Install the system binary first:
#   macOS:   brew install tesseract
#   Ubuntu:  sudo apt install tesseract-ocr
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki

pip install pytesseract Pillow
```

Then in `app.py`, swap `extract_text_easyocr` → `extract_text_tesseract` as
the primary in `extract_text_from_image()`.

---

### 3. Train the model

This downloads the UCI SMS Spam Collection (~500 KB) and saves the trained
artefacts to `model/`.

```bash
python train_model.py
```

Expected output:
```
📥  Downloading dataset …
✅  Dataset downloaded and extracted.
📂  Dataset loaded  : 5572 messages (747 spam, 4825 ham)

📊  Test Accuracy : 98.30%

              precision    recall  f1-score
        Ham       0.99      0.99      0.99
       Spam       0.96      0.95      0.96

💾  Model saved     → model/spam_model.joblib
💾  Vectorizer saved → model/vectorizer.joblib

🎉  Training complete! You can now run: python app.py
```

---

### 4. Start the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🔌 API Reference

### `POST /predict-text`

**Request (JSON):**
```json
{ "text": "Congratulations! You have won a FREE iPhone. Click now!" }
```

**Response:**
```json
{
  "label":          "spam",
  "confidence":     97.4,
  "spam_prob":      97.4,
  "ham_prob":       2.6,
  "extracted_text": "Congratulations! You have won a FREE iPhone. Click now!"
}
```

---

### `POST /predict-image`

**Request:** `multipart/form-data` with field `image` (any common image format).

**Response:** Same JSON shape as `/predict-text`, with `extracted_text`
containing the OCR output.

---

## 🧠 How the NLP Works

```
Raw message
    │
    ▼
clean_text()          # lowercase → strip URLs → remove non-alpha → trim
    │
    ▼
TfidfVectorizer       # unigrams + bigrams, 10k vocab, sublinear TF
    │
    ▼
MultinomialNB         # alpha=0.1 Laplace smoothing
    │
    ▼
predict_proba()       # [P(ham), P(spam)]
    │
    ▼
JSON response         # label + confidence + raw probabilities
```

**Why Multinomial NB + TF-IDF?**
- Naturally handles discrete, non-negative feature counts
- Extremely fast inference (< 1 ms per message)
- Competitive accuracy (~98%) on the UCI dataset
- TF-IDF down-weights common words and up-weights rare spam signals

---

## 🛠 Customisation Tips

| Goal | Change |
|---|---|
| Retrain on your own data | Modify `load_dataset()` in `train_model.py` |
| Switch to a stronger model | Replace `MultinomialNB` with `LogisticRegression` or `RandomForestClassifier` |
| Add email parsing | POST the email body text to `/predict-text` |
| Deploy to production | Use `gunicorn app:app` behind Nginx |

---

## 📜 License

MIT — free to use, modify, and distribute.
