"""Unit Tests for NLP Pipelines (Sentiment Analysis & Spam Detection)."""

import unittest
import numpy as np
from src.nlp.preprocessor import TextPreprocessor
from src.nlp.sentiment_model import SentimentClassifier
from src.nlp.spam_detector import SpamDetector
from src.nlp.pipeline import NLPPipeline


class TestNLPPipeline(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(max_vocab=1000, max_len=20)
        self.corpus = [
            "This product is amazing and brilliant!",
            "Terrible broken item, completely dissatisfied.",
            "Normal everyday text for indexing."
        ]
        self.preprocessor.fit(self.corpus)

    def test_text_cleaning(self):
        raw = "Check this OUT! https://example.com <b>Amazing</b>"
        cleaned = TextPreprocessor.clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertIn("amazing", cleaned)

    def test_transform_shape(self):
        transformed = self.preprocessor.transform(["This is a test"])
        self.assertEqual(transformed.shape, (1, 20))
        self.assertIsInstance(transformed, np.ndarray)

    def test_sentiment_prediction(self):
        clf = SentimentClassifier()
        pos_res = clf.predict("This product is brilliant and absolutely amazing!")
        neg_res = clf.predict("Completely broken, terrible, and useless waste of money.")
        
        self.assertEqual(pos_res["label"], "Positive")
        self.assertEqual(neg_res["label"], "Negative")
        self.assertGreater(pos_res["confidence"], 0.50)

    def test_spam_detection(self):
        spam_detector = SpamDetector()
        spam_res = spam_detector.predict("URGENT: You won $10,000 cash prize! Claim your free gift card now at http://win.xyz")
        ham_res = spam_detector.predict("Hey, are we still having lunch at 1 PM today? Let me know.")

        self.assertTrue(spam_res["is_spam"])
        self.assertFalse(ham_res["is_spam"])

    def test_unified_nlp_pipeline(self):
        pipeline = NLPPipeline()
        result = pipeline.process_all("Outstanding service, totally delighted!")
        self.assertIn("sentiment", result)
        self.assertIn("spam", result)
        self.assertEqual(result["sentiment"]["label"], "Positive")


if __name__ == "__main__":
    unittest.main()
