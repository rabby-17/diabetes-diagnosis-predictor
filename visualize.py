"""
Diabetes Data Visualization
This script creates visualizations for exploratory data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def plot_data_distribution(data):
    """Plot distribution of features by diabetes diagnosis"""
    print("Creating data distribution plots...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('Feature Distribution by Diabetes Diagnosis', fontsize=16, fontweight='bold')
    
    features = [col for col in data.columns if col != 'Outcome']
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 4, idx % 4]
        
        # Plot for each class
        data[data['Outcome'] == 0][feature].hist(ax=ax, alpha=0.6, label='No Diabetes', bins=30)
        data[data['Outcome'] == 1][feature].hist(ax=ax, alpha=0.6, label='Diabetes', bins=30)
        
        ax.set_title(feature, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/01_feature_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/01_feature_distribution.png")
    plt.close()

def plot_correlation_heatmap(data):
    """Plot correlation heatmap"""
    print("Creating correlation heatmap...")
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/02_correlation_heatmap.png")
    plt.close()

def plot_class_distribution(data):
    """Plot class distribution (imbalanced data check)"""
    print("Creating class distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    counts = data['Outcome'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(['No Diabetes', 'Diabetes'], counts.values, color=colors, alpha=0.7)
    axes[0].set_title('Class Distribution', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Class Distribution (%)', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/03_class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/03_class_distribution.png")
    plt.close()

def plot_model_performance(model, X_test, y_test, model_name):
    """Plot confusion matrix and ROC curve"""
    print(f"Creating performance plots for {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Model Performance', fontsize=14, fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Confusion Matrix', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[0].set_yticklabels(['No Diabetes', 'Diabetes'])
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontweight='bold')
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    filename = f'visualizations/04_performance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for Random Forest"""
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠ Feature importance not available for {model_name}")
        return
    
    print(f"Creating feature importance plot for {model_name}...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    plt.bar(range(len(feature_names)), importances[indices], color=colors, alpha=0.8)
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Importance', fontweight='bold')
    plt.title(f'{model_name} - Feature Importance', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/05_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/05_feature_importance.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("DIABETES DATA VISUALIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data = pd.read_csv('data/diabetes.csv')
    
    # Create visualizations
    plot_data_distribution(data)
    plot_correlation_heatmap(data)
    plot_class_distribution(data)
    
    # Load trained models for performance plots
    try:
        print("\nLoading trained models...")
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Prepare test data
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_scaled = scaler.transform(X_test)
        
        # Plot performance
        plot_model_performance(lr_model, X_test_scaled, y_test, 'Logistic Regression')
        plot_model_performance(rf_model, X_test_scaled, y_test, 'Random Forest')
        
        # Plot feature importance
        feature_names = X.columns.tolist()
        plot_feature_importance(rf_model, feature_names, 'Random Forest')
        
    except FileNotFoundError:
        print("⚠ Trained models not found. Please run train_model.py first.")
    
    print("\n" + "="*70)
    print("✓ All visualizations completed!")
    print("="*70)

if __name__ == "__main__":
    main()
