"""
Diabetes Risk Prediction - GUI Application
A Python GUI Application to Predict Diabetes Risk Using an AI Model
Built with Tkinter for user-friendly interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import os

class DiabetesRiskPredictorGUI:
    """
    GUI Application for Diabetes Risk Prediction
    Takes inputs: Glucose, Blood Pressure, BMI, Age, and other medical data
    Uses Pre-Trained Machine Learning Model for prediction
    """
    
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("Diabetes Risk Assessment System 🏥")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Configure style
        self.setup_styles()
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
        self.bg_color = "#f0f0f0"
        self.header_color = "#2c3e50"
        self.button_color = "#3498db"
        self.success_color = "#2ecc71"
        self.warning_color = "#e74c3c"
        
        # Configure root background
        self.root.configure(bg=self.bg_color)
    
    def load_model(self):
        """Load the pre-trained model and scaler"""
        try:
            self.model = joblib.load('models/random_forest_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.model_loaded = True
            print("✓ Model and scaler loaded successfully")
        except FileNotFoundError:
            self.model_loaded = False
            print("⚠ Model files not found. Please train the model first.")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Header Frame
        self.create_header()
        
        # Input Frame
        self.create_input_frame()
        
        # Button Frame
        self.create_button_frame()
        
        # Result Frame
        self.create_result_frame()
    
    def create_header(self):
        """Create header with title and description"""
        header_frame = tk.Frame(self.root, bg=self.header_color, height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Diabetes Risk Assessment System",
            font=("Helvetica", 24, "bold"),
            bg=self.header_color,
            fg="white"
        )
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Healthcare Prediction Model | Enter your medical data below",
            font=("Helvetica", 10),
            bg=self.header_color,
            fg="#ecf0f1"
        )
        subtitle_label.pack(pady=5)
    
    def create_input_frame(self):
        """Create input fields for medical data"""
        input_frame = ttk.Frame(self.root, padding="20")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        input_title = ttk.Label(
            input_frame,
            text="Medical Information",
            font=("Helvetica", 14, "bold")
        )
        input_title.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Input fields configuration
        self.fields = {
            'Pregnancies': {'row': 1, 'col': 0, 'range': (0, 17), 'unit': ''},
            'Glucose': {'row': 1, 'col': 1, 'range': (0, 200), 'unit': 'mg/dL'},
            'BloodPressure': {'row': 1, 'col': 2, 'range': (0, 122), 'unit': 'mm Hg'},
            'SkinThickness': {'row': 1, 'col': 3, 'range': (0, 99), 'unit': 'mm'},
            'Insulin': {'row': 2, 'col': 0, 'range': (0, 846), 'unit': 'mu U/ml'},
            'BMI': {'row': 2, 'col': 1, 'range': (0, 67.1), 'unit': 'kg/m²'},
            'DiabetesPedigreeFunction': {'row': 2, 'col': 2, 'range': (0, 2.42), 'unit': ''},
            'Age': {'row': 2, 'col': 3, 'range': (21, 81), 'unit': 'years'}
        }
        
        self.entries = {}
        
        # Create input fields
        for field_name, field_info in self.fields.items():
            row = field_info['row']
            col = field_info['col']
            unit = field_info['unit']
            
            # Label
            label = ttk.Label(
                input_frame,
                text=f"{field_name} {unit}".strip(),
                font=("Helvetica", 10)
            )
            label.grid(row=row, column=col, padx=5, pady=5, sticky=tk.W)
            
            # Entry
            entry = ttk.Entry(input_frame, width=15, font=("Helvetica", 10))
            entry.grid(row=row+1, column=col, padx=5, pady=5)
            self.entries[field_name] = entry
        
        # Note about ranges
        note_frame = ttk.Frame(input_frame)
        note_frame.grid(row=4, column=0, columnspan=4, pady=10)
        
        note_label = ttk.Label(
            note_frame,
            text="💡 Note: Enter numerical values. For normal inputs, try: Pregnancies=1, Glucose=120, BP=80, SkinThickness=20, Insulin=30, BMI=25, DPF=0.3, Age=30",
            font=("Helvetica", 9, "italic"),
            foreground="gray"
        )
        note_label.pack()
    
    def create_button_frame(self):
        """Create prediction and reset buttons"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Predict Button
        predict_btn = tk.Button(
            button_frame,
            text="🔮 Predict Diabetes Risk",
            command=self.predict,
            bg=self.button_color,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset Button
        reset_btn = tk.Button(
            button_frame,
            text="🔄 Clear All",
            command=self.reset_fields,
            bg="#95a5a6",
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
    
    def create_result_frame(self):
        """Create frame for displaying prediction results"""
        result_frame = ttk.LabelFrame(
            self.root,
            text="Prediction Result",
            padding="20"
        )
        result_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # Result text
        self.result_text = tk.Text(
            result_frame,
            height=8,
            width=80,
            font=("Courier", 11),
            bg="#f9f9f9",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    def predict(self):
        """Make prediction based on user input"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded. Please train the model first.")
            return
        
        try:
            # Get user inputs
            user_data = {}
            for field_name, entry in self.entries.items():
                value = entry.get().strip()
                if not value:
                    messagebox.showwarning("Input Error", f"Please enter {field_name}")
                    return
                try:
                    user_data[field_name] = float(value)
                except ValueError:
                    messagebox.showerror("Input Error", f"{field_name} must be a number")
                    return
            
            # Validate ranges
            for field_name, value in user_data.items():
                min_val, max_val = self.fields[field_name]['range']
                if not (min_val <= value <= max_val):
                    messagebox.showwarning(
                        "Range Warning",
                        f"{field_name} is outside typical range ({min_val}-{max_val})"
                    )
            
            # Create DataFrame for prediction
            df = pd.DataFrame([user_data])
            
            # Scale features
            df_scaled = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0][1]
            
            # Display results
            self.display_results(prediction, probability, user_data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def display_results(self, prediction, probability, user_data):
        """Display prediction results in the result frame"""
        # Determine diagnosis
        if prediction == 0:
            diagnosis = "NO DIABETES DETECTED ✓"
            color = self.success_color
            confidence = (1 - probability) * 100
            risk_level = "LOW RISK"
        else:
            diagnosis = "DIABETES DETECTED ⚠"
            color = self.warning_color
            confidence = probability * 100
            risk_level = "HIGH RISK"
        
        # Build result text
        result_text = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     DIABETES RISK PREDICTION REPORT                        ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 DIAGNOSIS: {diagnosis}
Risk Level: {risk_level}
Prediction Confidence: {confidence:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 INPUT VALUES:
  • Pregnancies:               {user_data['Pregnancies']:.0f}
  • Glucose Level:             {user_data['Glucose']:.0f} mg/dL
  • Blood Pressure:            {user_data['BloodPressure']:.0f} mm Hg
  • Skin Thickness:            {user_data['SkinThickness']:.0f} mm
  • Insulin Level:             {user_data['Insulin']:.0f} mu U/ml
  • BMI (Body Mass Index):     {user_data['BMI']:.2f} kg/m²
  • Diabetes Pedigree Function:{user_data['DiabetesPedigreeFunction']:.3f}
  • Age:                       {user_data['Age']:.0f} years

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚕️  MEDICAL ADVICE:
"""
        
        if prediction == 0:
            result_text += """
  ✓ Based on the AI model prediction, diabetes risk appears to be LOW.
  • Maintain a healthy lifestyle with regular exercise
  • Keep a balanced diet with controlled sugar intake
  • Regular health check-ups are recommended
  • Monitor glucose levels periodically
"""
        else:
            result_text += """
  ⚠ Based on the AI model prediction, diabetes risk appears to be HIGH.
  • Please consult with a healthcare professional immediately
  • Consider lifestyle modifications and diet changes
  • Regular glucose monitoring is strongly recommended
  • Follow medical advice for diabetes management
"""
        
        result_text += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 DISCLAIMER: This is an AI-powered prediction tool for educational purposes.
   It is NOT a substitute for professional medical diagnosis. Always consult
   with qualified healthcare professionals for accurate diagnosis and treatment.

╚════════════════════════════════════════════════════════════════════════════╝
"""
        
        # Update result text widget
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        self.result_text.config(state=tk.DISABLED)
    
    def reset_fields(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = DiabetesRiskPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
