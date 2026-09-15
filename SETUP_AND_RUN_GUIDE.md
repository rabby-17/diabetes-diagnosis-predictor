# Complete Setup and Run Guide 🚀

## Project Overview

This project has **4 main Python files** you need to run in a specific order:

```
1. train_model.py      ← Run FIRST (trains the AI model)
2. predict.py          ← Run SECOND (tests the model with sample data)
3. visualize.py        ← Run THIRD (creates charts and graphs)
4. gui_app.py          ← Run FOURTH (THE MAIN GUI APPLICATION) ⭐
```

---

## Step-by-Step Setup Instructions

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/rabby-17/diabetes-diagnosis-predictor.git
cd diabetes-diagnosis-predictor
```

Or download as ZIP and extract it.

---

### **Step 2: Create Virtual Environment** (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

This installs all required libraries:
- pandas (data handling)
- numpy (numerical computing)
- scikit-learn (machine learning)
- matplotlib & seaborn (visualization)
- joblib (model saving/loading)

---

### **Step 4: Download the Dataset**

1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Create a `data/` folder in your project
4. Place `diabetes.csv` inside the `data/` folder

Your folder structure should look like:
```
diabetes-diagnosis-predictor/
├── data/
│   └── diabetes.csv          ← Dataset goes here
├── models/                   ← (Created automatically after training)
├── visualizations/           ← (Created automatically after visualization)
├── train_model.py
├── predict.py
├── visualize.py
├── gui_app.py                ← THE MAIN FILE TO RUN ⭐
├── requirements.txt
└── README.md
```

---

## Running the Program

### **Phase 1: Train the Model** 🤖

Run this FIRST to train the AI model:

```bash
python train_model.py
```

**What happens:**
- Loads `diabetes.csv` from the `data/` folder
- Preprocesses the data
- Trains 2 models: Logistic Regression & Random Forest
- Saves models to `models/` folder
- Prints accuracy and other metrics

**Expected Output:**
```
======================================================================
DIABETES DIAGNOSIS PREDICTION - MODEL TRAINING
======================================================================

Loading data...
Data shape: (768, 9)

======================================================================
DATA PREPROCESSING
======================================================================
...
✓ Logistic Regression model trained successfully
✓ Random Forest model trained successfully

...MODEL COMPARISON SUMMARY
======================================================================

Logistic Regression - Accuracy: 0.7773, ROC AUC: 0.8407
Random Forest       - Accuracy: 0.7773, ROC AUC: 0.8462

✓ Best Model: Random Forest
```

**Files Created:**
- `models/logistic_regression_model.pkl`
- `models/random_forest_model.pkl`
- `models/scaler.pkl`

---

### **Phase 2: Test the Model** ✅

(Optional) Run this to test predictions:

```bash
python predict.py
```

**What happens:**
- Loads the trained model
- Makes predictions on sample patient data
- Shows prediction confidence and interpretation

**Expected Output:**
```
======================================================================
DIABETES DIAGNOSIS PREDICTION - INFERENCE
======================================================================

✓ Model loaded from: models/random_forest_model.pkl
✓ Scaler loaded from: models/scaler.pkl

----------------------------------------------------------------------
EXAMPLE 1: Single Patient Prediction
----------------------------------------------------------------------

Patient Data: {'Pregnancies': 6, 'Glucose': 148, ...}
Prediction: 1
Probability of Diabetes: 0.7823
Interpretation: ⚠ Diabetes Detected (Confidence: 78.23%)
```

---

### **Phase 3: Generate Visualizations** 📊

(Optional) Run this to create charts:

```bash
python visualize.py
```

**What happens:**
- Creates 5 visualization PNG files
- Shows feature distributions
- Displays correlation heatmap
- Shows model performance metrics

**Files Created in `visualizations/` folder:**
1. `01_feature_distribution.png` - Feature histograms
2. `02_correlation_heatmap.png` - Feature correlations
3. `03_class_distribution.png` - Dataset balance
4. `04_performance_*.png` - Confusion matrices & ROC curves
5. `05_feature_importance.png` - Important features

---

### **Phase 4: Run the GUI Application** 🎯 **← THE MAIN FILE!**

This is the **MAIN APPLICATION** - Run this to use the diabetes predictor:

```bash
python gui_app.py
```

**What happens:**
- A beautiful GUI window opens
- Enter patient medical data
- Click "🔮 Predict Diabetes Risk"
- Get instant prediction with confidence score
- See medical recommendations

---

## GUI Application Usage

### **Step-by-Step to Use the GUI:**

1. **Start the application:**
   ```bash
   python gui_app.py
   ```

2. **Enter Medical Data:**
   - **Pregnancies**: Number of pregnancies (0-17)
   - **Glucose**: Blood glucose level (0-200 mg/dL)
   - **BloodPressure**: Diastolic BP (0-122 mm Hg)
   - **SkinThickness**: Skin fold thickness (0-99 mm)
   - **Insulin**: Serum insulin level (0-846 mu U/ml)
   - **BMI**: Body mass index (0-67.1 kg/m²)
   - **DiabetesPedigreeFunction**: Family diabetes history (0-2.42)
   - **Age**: Age in years (21-81)

3. **Example Values (Healthy Person):**
   ```
   Pregnancies: 1
   Glucose: 100
   BloodPressure: 80
   SkinThickness: 20
   Insulin: 30
   BMI: 25
   DiabetesPedigreeFunction: 0.3
   Age: 30
   ```

4. **Click "🔮 Predict Diabetes Risk"**

5. **Get Results:**
   - ✓ No Diabetes Detected (Low Risk)
   - ⚠ Diabetes Detected (High Risk)
   - Confidence percentage
   - Medical recommendations

6. **Click "🔄 Clear All"** to reset and make another prediction

---

## Quick Cheat Sheet ⚡

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `pip install -r requirements.txt` | Install dependencies |
| 2 | `python train_model.py` | Train the AI model |
| 3 | `python predict.py` | Test model (optional) |
| 4 | `python visualize.py` | Create charts (optional) |
| 5 | `python gui_app.py` | **RUN THE GUI APP** ⭐ |

---

## Troubleshooting

### **Error: "Model files not found"**
- **Solution**: Run `python train_model.py` first
- Make sure you're in the correct directory

### **Error: "diabetes.csv not found"**
- **Solution**: Download the dataset and place it in `data/` folder
- Download link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### **Error: "ModuleNotFoundError: No module named 'sklearn'"**
- **Solution**: Run `pip install -r requirements.txt`
- Make sure your virtual environment is activated

### **GUI won't open**
- **Solution**: Make sure you have Tkinter installed
  - **Windows**: Usually included with Python
  - **Linux**: `sudo apt-get install python3-tk`
  - **Mac**: Usually included with Python

### **Port/Model already in use**
- **Solution**: Close other Python windows and try again

---

## Project Structure

```
diabetes-diagnosis-predictor/
│
├── data/
│   └── diabetes.csv                 ← Download and place here
│
├── models/
│   ├── logistic_regression_model.pkl  (created after training)
│   ├── random_forest_model.pkl        (created after training)
│   └── scaler.pkl                     (created after training)
│
├── visualizations/                   (created after visualization)
│   ├── 01_feature_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_class_distribution.png
│   ├── 04_performance_*.png
│   └── 05_feature_importance.png
│
├── train_model.py                   ← Step 1: Train Model
├── predict.py                       ← Step 2: Test Model (Optional)
├── visualize.py                     ← Step 3: Generate Charts (Optional)
├── gui_app.py                       ← Step 4: MAIN GUI APPLICATION ⭐
├── requirements.txt                 ← Dependencies
├── .gitignore
└── README.md
```

---

## Expected Outcomes

✅ Model Training Accuracy: ~77-78%
✅ ROC-AUC Score: ~0.84-0.86
✅ Real-time GUI predictions with <1 second latency
✅ Professional medical disclaimer included
✅ Comprehensive error handling
✅ Detailed logging to `diabetes_app.log`

---

## Important Notes ⚠️

1. **Medical Disclaimer**: This is an educational tool, NOT for actual medical diagnosis
2. **Always consult healthcare professionals** for real medical decisions
3. **Error Logging**: Check `diabetes_app.log` for troubleshooting
4. **Dataset**: Dataset is imbalanced (~65% No Diabetes, ~35% Diabetes)

---

## File Descriptions

| File | Purpose |
|------|---------|
| `train_model.py` | Trains ML models from scratch |
| `predict.py` | Tests predictions with sample data |
| `visualize.py` | Creates data visualizations |
| `gui_app.py` | **Main GUI application** ⭐ |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `README.md` | Project overview |

---

## Contact & Support

For issues or questions:
1. Check `diabetes_app.log` for error details
2. Verify dataset is in `data/` folder
3. Ensure all dependencies are installed
4. Make sure models are trained before running GUI

---

**Created by rabby-17** | Educational Purpose Only
