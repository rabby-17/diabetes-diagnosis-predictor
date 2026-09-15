"""
Diabetes Diagnosis Prediction - Inference Script
This script loads a trained model and makes predictions on new data.
"""

import joblib
import pandas as pd
import numpy as np
import sys

class DiabetesPredictor:
    """
    A class to handle diabetes diagnosis predictions
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize the predictor with a trained model and scaler
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")
    
    def predict_single(self, features):
        """
        Make a prediction for a single patient
        
        Args:
            features: List or dict of patient features
            
        Returns:
            prediction: 0 (No Diabetes) or 1 (Diabetes)
            probability: Probability of having diabetes
        """
        # Convert to DataFrame if dict
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = pd.DataFrame([features], columns=self.feature_names)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return prediction, probability
    
    def predict_batch(self, data):
        """
        Make predictions for multiple patients
        
        Args:
            data: DataFrame with patient data
            
        Returns:
            DataFrame with predictions and probabilities
        """
        # Scale features
        features_scaled = self.scaler.transform(data)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Create result DataFrame
        result = data.copy()
        result['Prediction'] = predictions
        result['Diabetes_Probability'] = probabilities
        result['Diagnosis'] = result['Prediction'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        return result
    
    def interpret_result(self, prediction, probability):
        """
        Provide human-readable interpretation of prediction
        
        Args:
            prediction: Model prediction (0 or 1)
            probability: Probability of diabetes
            
        Returns:
            str: Interpretation message
        """
        if prediction == 0:
            return f"✓ No Diabetes Detected (Confidence: {(1-probability)*100:.2f}%)"
        else:
            return f"⚠ Diabetes Detected (Confidence: {probability*100:.2f}%)"

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
def main():
    print("\n" + "="*70)
    print("DIABETES DIAGNOSIS PREDICTION - INFERENCE")
    print("="*70)
    
    # Initialize predictor
    predictor = DiabetesPredictor(
        'models/random_forest_model.pkl',
        'models/scaler.pkl'
    )
    
    # Example 1: Single patient prediction
    print("\n" + "-"*70)
    print("EXAMPLE 1: Single Patient Prediction")
    print("-"*70)
    
    patient_1 = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    prediction, probability = predictor.predict_single(patient_1)
    print(f"\nPatient Data: {patient_1}")
    print(f"Prediction: {prediction}")
    print(f"Probability of Diabetes: {probability:.4f}")
    print(f"Interpretation: {predictor.interpret_result(prediction, probability)}")
    
    # Example 2: Another patient
    print("\n" + "-"*70)
    print("EXAMPLE 2: Another Patient")
    print("-"*70)
    
    patient_2 = {
        'Pregnancies': 1,
        'Glucose': 85,
        'BloodPressure': 66,
        'SkinThickness': 29,
        'Insulin': 0,
        'BMI': 26.6,
        'DiabetesPedigreeFunction': 0.351,
        'Age': 31
    }
    
    prediction, probability = predictor.predict_single(patient_2)
    print(f"\nPatient Data: {patient_2}")
    print(f"Prediction: {prediction}")
    print(f"Probability of Diabetes: {probability:.4f}")
    print(f"Interpretation: {predictor.interpret_result(prediction, probability)}")
    
    # Example 3: Batch prediction from test set
    print("\n" + "-"*70)
    print("EXAMPLE 3: Batch Prediction (First 5 test samples)")
    print("-"*70)
    
    # Load test data
    data = pd.read_csv('data/diabetes.csv')
    X = data.drop('Outcome', axis=1)
    test_data = X.head(5)
    
    results = predictor.predict_batch(test_data)
    print(f"\n{results[['Glucose', 'BMI', 'Age', 'Diagnosis', 'Diabetes_Probability']]}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
