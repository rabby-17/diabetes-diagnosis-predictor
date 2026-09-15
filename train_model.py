"""
Diabetes Risk Prediction - Simple Model Training (No scipy)
Uses basic Python and NumPy to avoid DLL issues
Compatible with Python 3.14
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleLogisticRegression:
    """Simple Logistic Regression implementation without scipy"""
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X, y):
        """Train the model"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Backward pass
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights) + self.bias
        proba = self.sigmoid(z)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        """Predict class"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SimpleRandomForest:
    """Simple Decision Tree for Random Forest"""
    
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        """Train the tree"""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build decision tree"""
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(X) < 2:
            return {'value': np.bincount(y).argmax()}
        
        n_features = X.shape[1]
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Find best split
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds[::max(1, len(thresholds)//10)]:  # Sample thresholds
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                # Information gain
                parent_entropy = self._entropy(y)
                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[right_mask])
                
                gain = parent_entropy - (len(y[left_mask]) / len(y) * left_entropy + 
                                        len(y[right_mask]) / len(y) * right_entropy)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            return {'value': np.bincount(y).argmax()}
        
        left_mask = X[:, best_feature] <= best_threshold
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }
    
    def _entropy(self, y):
        """Calculate entropy"""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    
    def predict(self, X):
        """Make predictions"""
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree for single sample"""
        if 'value' in node:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])
    
    def predict_proba(self, X):
        """Predict probabilities"""
        predictions = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[predictions == 0, 0] = 1
        proba[predictions == 1, 1] = 1
        return proba


def load_data(filepath):
    """Load and validate the diabetes dataset"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        data = pd.read_csv(filepath)
        logger.info(f"✓ Data loaded successfully: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    """Preprocess the diabetes data"""
    try:
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Check for any remaining NaN
        if data.isnull().any().any():
            logger.warning("NaN values found, filling with mean")
            data = data.fillna(data.mean())
        
        # Separate features and target
        if data.shape[1] == 9:  # Standard diabetes dataset
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        else:
            raise ValueError(f"Expected 9 columns, got {data.shape[1]}")
        
        logger.info(f"✓ Data preprocessed: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data and scale features manually"""
    try:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Scale features manually (standardization)
        class SimpleScaler:
            def __init__(self):
                self.mean = None
                self.std = None
            
            def fit_transform(self, X):
                self.mean = np.mean(X, axis=0)
                self.std = np.std(X, axis=0) + 1e-8
                return (X - self.mean) / self.std
            
            def transform(self, X):
                return (X - self.mean) / self.std
        
        scaler = SimpleScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"✓ Data split: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    except Exception as e:
        logger.error(f"Error splitting/scaling data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics manually
        accuracy = np.mean(y_pred == y_test)
        
        # ROC-AUC calculation
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {str(e)}")
        raise

def save_model(model, scaler, model_name, model_path='models'):
    """Save trained model and scaler"""
    try:
        # Create models directory if it doesn't exist
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = f"{model_path}/{model_name}_model.pkl"
        joblib.dump(model, model_file)
        logger.info(f"✓ Model saved: {model_file}")
        
        # Save scaler (only once)
        if model_name == "random_forest":
            scaler_file = f"{model_path}/scaler.pkl"
            joblib.dump(scaler, scaler_file)
            logger.info(f"✓ Scaler saved: {scaler_file}")
        
        return model_file
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    """Main training pipeline"""
    try:
        print("=" * 70)
        print("DIABETES DIAGNOSIS PREDICTION - MODEL TRAINING")
        print("Python 3.14 Compatible (No scipy DLL issues)")
        print("=" * 70)
        
        # Load data
        print("\n📊 Loading dataset...")
        data = load_data('data/diabetes.csv')
        print(f"   Dataset shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        
        # Preprocess
        print("\n🔧 Preprocessing data...")
        X, y = preprocess_data(data)
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Class distribution: {np.bincount(y.astype(int))}")
        
        # Split and scale
        print("\n✂️  Splitting and scaling data...")
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
        
        # Train models
        print("\n🤖 Training models...")
        print("   Training Logistic Regression...")
        lr_model = SimpleLogisticRegression(learning_rate=0.01, iterations=1000)
        lr_model.fit(X_train, y_train.astype(int))
        print("   ✓ Logistic Regression trained")
        
        print("   Training Random Forest...")
        rf_model = SimpleRandomForest(max_depth=10)
        rf_model.fit(X_train, y_train.astype(int))
        print("   ✓ Random Forest trained")
        
        # Evaluate models
        print("\n📈 Evaluating models...")
        lr_metrics = evaluate_model(lr_model, X_test, y_test.astype(int), "Logistic Regression")
        rf_metrics = evaluate_model(rf_model, X_test, y_test.astype(int), "Random Forest")
        
        # Save models
        print("\n💾 Saving models...")
        save_model(lr_model, scaler, "logistic_regression")
        save_model(rf_model, scaler, "random_forest")
        
        # Compare models
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nLogistic Regression:")
        print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
        
        print(f"\nRandom Forest:")
        print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
        
        best_model = "Random Forest" if rf_metrics['accuracy'] > lr_metrics['accuracy'] else "Logistic Regression"
        print(f"\n✓ Best Model: {best_model}")
        print("\n✓ Training completed successfully!")
        print(f"✓ Models saved to 'models/' folder")
        print(f"✓ You can now run: python gui_app.py")
        print("=" * 70)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in training: {str(e)}\n{traceback.format_exc()}")
        print(f"\n❌ ERROR: {str(e)}")
        print("\nMake sure:")
        print("  1. data/diabetes.csv exists")
        print("  2. You're running Python 3.8 or higher")
        raise

if __name__ == "__main__":
    main()
