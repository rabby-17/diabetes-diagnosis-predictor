# Diabetes Diagnosis Predictor 🏥

A machine learning application to predict diabetes diagnosis using Python. This project includes model training, evaluation, and prediction capabilities using logistic regression and random forest algorithms.

## 📋 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)

## ✨ Features

- **Data Preprocessing**: Handles missing values, feature scaling, train-test splitting
- **Multiple Models**: Logistic Regression & Random Forest classifiers
- **Model Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Predictions**: Single patient and batch prediction
- **Visualizations**: EDA plots, confusion matrices, ROC curves
- **Model Persistence**: Save/load models with joblib

## 📦 Requirements

Python 3.7+

```bash
pip install -r requirements.txt
```

## 🚀 Installation

```bash
git clone https://github.com/rabby-17/diabetes-diagnosis-predictor.git
cd diabetes-diagnosis-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📊 Dataset

Uses **Pima Indians Diabetes Dataset** from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

**Target**: Outcome (0=No Diabetes, 1=Diabetes)

## 🎯 Quick Start

### 1. Download Dataset
- Get `diabetes.csv` from Kaggle
- Place in `data/` directory
- Create directories: `mkdir -p data models visualizations`

### 2. Train Models
```bash
python train_model.py
```

### 3. Make Predictions
```bash
python predict.py
```

### 4. Generate Visualizations
```bash
python visualize.py
```

## 📁 Project Structure

```
diabetes-diagnosis-predictor/
├── data/diabetes.csv
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── visualizations/
├── train_model.py
├── predict.py
├── visualize.py
└── requirements.txt
```

## 📈 Model Performance

**Expected Accuracy**: 75-78%  
**Expected ROC-AUC**: 0.84-0.86

## 📝 License

MIT License - Open source

---

**Created by rabby-17** | For educational purposes only
