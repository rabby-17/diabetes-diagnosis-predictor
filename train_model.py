"""
Diabetes Diagnosis Prediction Model - Training Script
This script loads the diabetes dataset, preprocesses it, trains a model, and saves it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_data(filepath):
    """Load the diabetes dataset"""
    print("Loading data...")
    data = pd.read_csv(filepath)
    print(f"Data shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    return data

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
def preprocess_data(data):
    """
    Preprocess the diabetes data
    - Handle missing values
    - Separate features and target
    - Return preprocessed data
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Check for missing values
    print(f"\nMissing values:\n{data.isnull().sum()}")
    
    # Display basic statistics
    print(f"\nData Statistics:\n{data.describe()}")
    
    # Separate features (X) and target (y)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    print(f"\nTarget variable distribution:\n{y.value_counts()}")
    print(f"Class 0 (No Diabetes): {(y==0).sum()}")
    print(f"Class 1 (Diabetes): {(y==1).sum()}")
    
    return X, y

# ============================================================================
# 3. TRAIN-TEST SPLIT AND SCALING
# ============================================================================
def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets, then scale features
    """
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT & SCALING")
    print("="*70)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*70)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("✓ Logistic Regression model trained successfully")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("✓ Random Forest model trained successfully")
    return model

# ============================================================================
# 5. EVALUATE MODELS
# ============================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print("\n" + "-"*70)
    print(f"EVALUATION - {model_name}")
    print("-"*70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return accuracy, roc_auc

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("DIABETES DIAGNOSIS PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    data = load_data('data/diabetes.csv')
    
    # Preprocess
    X, y = preprocess_data(data)
    
    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate models
    lr_acc, lr_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_acc, rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    print("✓ Logistic Regression model saved")
    
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("✓ Random Forest model saved")
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Scaler saved")
    
    # Summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\nLogistic Regression - Accuracy: {lr_acc:.4f}, ROC AUC: {lr_auc:.4f}")
    print(f"Random Forest       - Accuracy: {rf_acc:.4f}, ROC AUC: {rf_auc:.4f}")
    
    best_model = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"
    print(f"\n✓ Best Model: {best_model}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
