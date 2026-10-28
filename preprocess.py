"""
preprocess.py
-------------
Shared text preprocessing module for SpamGuard AI.
Used identically during model training (train_model.py) and live inference (app.py).
"""

import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """
    Clean raw text and normalize spam-indicative tokens.

    Transformations:
      1. Replace URLs (http/https/www) -> ' URL '
      2. Replace symbols before character stripping:
         - '$' -> ' DOLLAR '
         - '₹' -> ' RUPEE '
         - '%' -> ' PERCENT '
         - '!' -> ' EXCLAMATION '
      3. Replace phone numbers (10+ digits) -> ' PHONE '
      4. Replace remaining numbers -> ' NUMBER '
      5. Strip unsupported non-alphanumeric punctuation
      6. Normalize whitespace and convert to lowercase for TF-IDF tokenization.

    Parameters:
        text (str): Raw input text or message string.

    Returns:
        str: Normalized, tokenized clean string.
    """
    if text is None or not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    # 1. URLs
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # 2. Key spam-signal symbols (converted to distinct tokens)
    text = re.sub(r"\$", " DOLLAR ", text)
    text = re.sub(r"₹", " RUPEE ", text)
    text = re.sub(r"%", " PERCENT ", text)
    text = re.sub(r"!+", " EXCLAMATION ", text)

    # 3. Phone numbers (10 or more contiguous digits)
    text = re.sub(r"\b\d{10,}\b", " PHONE ", text)

    # 4. Remaining numeric sequences
    text = re.sub(r"\d+", " NUMBER ", text)

    # 5. Remove unsupported characters, retaining alphanumeric tokens & whitespace
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 6. Normalize whitespace and lowercase so tokens match TF-IDF vocabulary
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text
