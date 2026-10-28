"""
train_model.py
--------------
Downloads the UCI SMS Spam Collection dataset, trains a Multinomial Naive Bayes
classifier with TF-IDF vectorization, and saves both the model and vectorizer
to disk using joblib.

Run this script once before starting the Flask app:
    python train_model.py
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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


# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR     = "data"
DATASET_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")


def download_dataset() -> None:
    """Download and unzip the UCI SMS Spam Collection dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")

    if not os.path.exists(DATASET_FILE):
        print("📥  Downloading dataset …")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(zip_path)
        print("✅  Dataset downloaded and extracted.")
    else:
        print("✅  Dataset already present – skipping download.")


def load_dataset() -> pd.DataFrame:
    """Read the tab-separated SMS Spam Collection into a DataFrame."""
    df = pd.read_csv(
        DATASET_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    # Binary encode labels: spam → 1, ham → 0
    df["label_enc"] = df["label"].map({"spam": 1, "ham": 0})
    df["clean_message"] = df["message"].apply(clean_text)
    return df


def train_and_save_model(df: pd.DataFrame) -> dict:
    """
    Train a TF-IDF + Multinomial Naive Bayes pipeline.

    Why Multinomial NB?
    • Naturally suited to text classification with discrete, non-negative features.
    • Fast to train even on large corpora.
    • Competitive accuracy on spam detection tasks with low latency.

    Why TF-IDF instead of raw counts?
    • Down-weights words appearing in nearly every message.
    • Up-weights rare but characteristic spam tokens (e.g. 'dollar', 'rupee', 'free').
    """
    X = df["clean_message"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Vectorizer ────────────────────────────────────────────────────────────
    # ngram_range=(1, 2) captures unigrams AND bigrams ("free dollar", "rupee win")
    # sublinear_tf=True applies 1 + log(tf) scaling to dampen frequent terms
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
    )

    # ── Classifier ────────────────────────────────────────────────────────────
    # alpha=0.1 (Laplace smoothing) works better than default 1.0 for sparse spam text
    model = MultinomialNB(alpha=0.1)

    # Fit vectorizer on training data, transform both splits
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

    print(f"\n📊  Test Accuracy : {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Ham: {cm[0][0]} | False Spam: {cm[0][1]}")
    print(f"  False Ham: {cm[1][0]} | True Spam: {cm[1][1]}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # ── Persist artifacts ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"💾  Model saved     → {MODEL_PATH}")
    print(f"💾  Vectorizer saved → {VECTORIZER_PATH}")

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def main() -> None:
    download_dataset()
    df = load_dataset()
    print(f"\n📂  Dataset loaded  : {len(df)} messages "
          f"({df['label'].value_counts()['spam']} spam, "
          f"{df['label'].value_counts()['ham']} ham)")
    train_and_save_model(df)
    print("\n🎉  Training complete! You can now run: python app.py")


if __name__ == "__main__":
    main()
