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
import re
import urllib.request
import zipfile
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


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


def train_and_save_model(df: pd.DataFrame) -> None:
    """
    Train a TF-IDF + Multinomial Naive Bayes pipeline.

    Why Multinomial NB?
    • Naturally suited to text classification with discrete, non-negative features.
    • Fast to train even on large corpora.
    • Surprisingly competitive accuracy on spam detection tasks.

    Why TF-IDF instead of raw counts?
    • Down-weights words that appear in nearly every message (e.g., "the", "is"),
      which have low discriminative power.
    • Up-weights rare but characteristic spam tokens (e.g., "free", "winner").
    """
    X = df["clean_message"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Vectorizer ────────────────────────────────────────────────────────────
    # ngram_range=(1, 2)  captures both unigrams AND bigrams ("free prize")
    # max_features limits vocabulary size to prevent overfitting on noise
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,      # apply log(1 + tf) to dampen high-frequency terms
    )

    # ── Classifier ────────────────────────────────────────────────────────────
    # alpha=0.1 (Laplace smoothing) works better than the default 1.0 here
    model = MultinomialNB(alpha=0.1)

    # Fit vectorizer on training data, transform both splits
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n📊  Test Accuracy : {acc * 100:.2f}%")
    print("\n" + classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # ── Persist artifacts ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"💾  Model saved     → {MODEL_PATH}")
    print(f"💾  Vectorizer saved → {VECTORIZER_PATH}")


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
