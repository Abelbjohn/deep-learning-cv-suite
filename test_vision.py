"""Unit Tests for Computer Vision Suite (CNN Detection, 68 Landmarks & 30+ FPS Processor)."""

import unittest
import numpy as np
from src.vision.object_detector import ObjectDetector
from src.vision.landmark_tracker import FacialLandmarkTracker
from src.vision.stream_processor import StreamProcessor


class TestComputerVisionSuite(unittest.TestCase):

    def setUp(self):
        self.detector = ObjectDetector()
        self.tracker = FacialLandmarkTracker()
        self.processor = StreamProcessor(mode="both")
        self.synthetic_frame = self.processor.generate_synthetic_frame()

    def test_object_detection(self):
        detections = self.detector.detect(self.synthetic_frame)
        self.assertIsInstance(detections, list)
        if detections:
            first = detections[0]
            self.assertIn("class", first)
            self.assertIn("confidence", first)
            self.assertIn("box", first)
            self.assertEqual(len(first["box"]), 4)

    def test_68_facial_landmarks(self):
        box = (100, 100, 200, 250)
        landmarks = self.tracker.estimate_68_landmarks(box)
        self.assertEqual(landmarks.shape, (68, 2))
        
        # EAR and MAR computation
        right_eye = landmarks[36:42]
        mouth = landmarks[48:60]
        ear = self.tracker.compute_ear(right_eye)
        mar = self.tracker.compute_mar(mouth)

        self.assertGreater(ear, 0.0)
        self.assertGreater(mar, 0.0)

    def test_stream_processor_throughput(self):
        annotated, metrics = self.processor.process_frame(self.synthetic_frame)
        self.assertEqual(annotated.shape, self.synthetic_frame.shape)
        self.assertIn("fps", metrics)
        self.assertIn("latency_ms", metrics)
        self.assertGreater(metrics["fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
