"""
tests/test_spamguard.py
-----------------------
Automated test suite for SpamGuard AI.
Tests preprocessing, model inference, Flask API endpoints, OCR caching, and assets.
"""

import io
import os
import unittest
from PIL import Image, ImageDraw

import app
import preprocess


class TestPreprocessing(unittest.TestCase):
    """Test text cleaning and symbol-to-token normalization."""

    def test_symbols_conversion(self):
        text = "Win $500 or ₹40,000 with 100% bonus! Claim now!"
        cleaned = preprocess.clean_text(text)
        self.assertIn("dollar", cleaned)
        self.assertIn("rupee", cleaned)
        self.assertIn("percent", cleaned)
        self.assertIn("exclamation", cleaned)

    def test_url_and_phone_normalization(self):
        text = "Visit https://spam-deal.xyz or call 9876543210 immediately!"
        cleaned = preprocess.clean_text(text)
        self.assertIn("url", cleaned)
        self.assertIn("phone", cleaned)

    def test_empty_and_none_input(self):
        self.assertEqual(preprocess.clean_text(""), "")
        self.assertEqual(preprocess.clean_text("   "), "")
        self.assertEqual(preprocess.clean_text(None), "")
        self.assertEqual(preprocess.clean_text(12345), "")

    def test_ordinary_legitimate_message(self):
        text = "Hey are we still meeting for lunch at 12:30?"
        cleaned = preprocess.clean_text(text)
        self.assertIn("hey", cleaned)
        self.assertIn("meeting", cleaned)
        self.assertIn("lunch", cleaned)
        self.assertIn("number", cleaned)


class TestModelInference(unittest.TestCase):
    """Test model & vectorizer classification logic."""

    def test_model_loaded(self):
        self.assertIsNotNone(app.model, "ML model should be loaded.")
        self.assertIsNotNone(app.vectorizer, "TF-IDF vectorizer should be loaded.")

    def test_spam_prediction(self):
        spam_msg = "Congratulations! You have won a $1,000 cash prize! Claim 100% free now!"
        res = app.classify(spam_msg)
        self.assertEqual(res["label"], "spam")
        self.assertGreater(res["spam_prob"], 50.0)
        self.assertAlmostEqual(res["spam_prob"] + res["ham_prob"], 100.0, places=1)

    def test_ham_prediction(self):
        ham_msg = "Hey, are you free this afternoon to review the notes for class?"
        res = app.classify(ham_msg)
        self.assertEqual(res["label"], "ham")
        self.assertGreater(res["ham_prob"], 50.0)
        self.assertAlmostEqual(res["spam_prob"] + res["ham_prob"], 100.0, places=1)


class TestFlaskEndpoints(unittest.TestCase):
    """Test Flask HTTP routes and contract compliance."""

    def setUp(self):
        app.app.config["TESTING"] = True
        self.client = app.app.test_client()

    def test_homepage_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("SpamGuard", html)
        self.assertIn("Text Analysis", html)
        self.assertIn("Image OCR Scan", html)

    def test_predict_text_valid_spam(self):
        payload = {"text": "URGENT! You won $5000 prize. Call 9876543210 to claim free!"}
        res = self.client.post("/predict-text", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("label", data)
        self.assertIn("confidence", data)
        self.assertIn("spam_prob", data)
        self.assertIn("ham_prob", data)
        self.assertIn("extracted_text", data)
        self.assertEqual(data["label"], "spam")

    def test_predict_text_valid_ham(self):
        payload = {"text": "I will be home around 6 pm, see you soon."}
        res = self.client.post("/predict-text", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["label"], "ham")

    def test_predict_text_empty(self):
        res = self.client.post("/predict-text", json={"text": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_predict_text_whitespace_only(self):
        res = self.client.post("/predict-text", json={"text": "   \n\t  "})
        self.assertEqual(res.status_code, 400)

    def test_predict_text_missing_field(self):
        res = self.client.post("/predict-text", json={"wrong_key": "hello"})
        self.assertEqual(res.status_code, 400)

    def test_predict_image_unsupported_extension(self):
        data = {
            "image": (io.BytesIO(b"fake pdf content"), "malicious.pdf")
        }
        res = self.client.post("/predict-image", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 415)
        self.assertIn("Unsupported file type", res.get_json()["error"])

    def test_predict_image_blank_image(self):
        # Create a blank white image with no text
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        data = {"image": (buf, "blank.png")}
        res = self.client.post("/predict-image", data=data, content_type="multipart/form-data")
        # Blank image has no text extracted -> 422 Unprocessable Entity
        self.assertEqual(res.status_code, 422)
        self.assertIn("Could not extract any text", res.get_json()["error"])


class TestEasyOCRSingleton(unittest.TestCase):
    """Verify that EasyOCR Reader is cached as a singleton and reused across calls."""

    def test_reader_is_cached_singleton(self):
        reader1 = app.get_easyocr_reader()
        reader2 = app.get_easyocr_reader()
        self.assertIs(reader1, reader2, "get_easyocr_reader() must return the same cached instance.")


class TestAssetsAndDocumentation(unittest.TestCase):
    """Verify integrity of files, directories, and documentation assets."""

    def test_templates_directory_exists(self):
        self.assertTrue(os.path.isdir("templates"), "templates must be a directory.")
        self.assertTrue(
            os.path.isfile(os.path.join("templates", "index.html")),
            "templates/index.html must exist as a file.",
        )

    def test_screenshots_exist(self):
        for i in range(1, 5):
            path = os.path.join("Screenshot", f"Picture{i}.png")
            self.assertTrue(
                os.path.isfile(path),
                f"Screenshot {path} must exist on disk.",
            )

    def test_gitignore_exists(self):
        self.assertTrue(os.path.isfile(".gitignore"), ".gitignore must exist.")
        with open(".gitignore", "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("venv/", content)
            self.assertIn("__pycache__/", content)


if __name__ == "__main__":
    unittest.main()
