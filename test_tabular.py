"""Unit Tests for Tabular Models (Diabetes Risk & Bike Demand)."""

import unittest
import pandas as pd
from src.tabular.diabetes_predictor import DiabetesPredictor
from src.tabular.bike_demand_predictor import BikeDemandPredictor


class TestTabularModels(unittest.TestCase):

    def setUp(self):
        self.diabetes_predictor = DiabetesPredictor()
        self.bike_predictor = BikeDemandPredictor()

    def test_diabetes_patient_prediction(self):
        high_risk_patient = {
            "Glucose": 175.0, "BMI": 38.5, "Age": 55, "BloodPressure": 90,
            "Insulin": 220, "Pregnancies": 5, "SkinThickness": 32, "DiabetesPedigreeFunction": 0.95
        }
        res = self.diabetes_predictor.predict_patient(high_risk_patient)
        self.assertIn("risk_probability", res)
        self.assertIn("risk_tier", res)
        self.assertGreater(res["risk_probability"], 0.50)

    def test_bike_demand_prediction(self):
        rush_hour_conditions = {
            "hr": 17, "workingday": 1, "temp": 0.65,
            "hum": 0.40, "weathersit": 1, "windspeed": 0.15
        }
        res = self.bike_predictor.predict_demand(rush_hour_conditions)
        self.assertIn("predicted_count", res)
        self.assertIn("confidence_interval", res)
        self.assertIn("demand_tier", res)
        self.assertGreater(res["predicted_count"], 0)


if __name__ == "__main__":
    unittest.main()
