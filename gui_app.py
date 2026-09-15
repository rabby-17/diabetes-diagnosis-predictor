"""
Diabetes Risk Prediction - GUI Application with Enhanced Error Handling
A Python GUI Application to Predict Diabetes Risk Using an AI Model
Built with Tkinter for user-friendly interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import joblib
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiabetesRiskPredictorGUI:
    """
    GUI Application for Diabetes Risk Prediction
    Takes inputs: Glucose, Blood Pressure, BMI, Age, and other medical data
    Uses Pre-Trained Machine Learning Model for prediction
    Includes comprehensive error handling
    """
    
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("Diabetes Risk Assessment System 🏥")
        self.root.geometry("900x750")
        self.root.resizable(False, False)
        
        # Configure style
        self.setup_styles()
        
        # Load model
        self.load_model()
        
        # Create GUI
        try:
            self.create_widgets()
            logger.info("GUI created successfully")
        except Exception as e:
            logger.error(f"Error creating GUI: {str(e)}")
            messagebox.showerror("GUI Error", f"Failed to create GUI: {str(e)}")
    
    def setup_styles(self):
        """Configure ttk styles"""
        try:
            style = ttk.Style()
            style.theme_use('clam')
            
            # Colors
            self.bg_color = "#f0f0f0"
            self.header_color = "#2c3e50"
            self.button_color = "#3498db"
            self.success_color = "#2ecc71"
            self.warning_color = "#e74c3c"
            self.error_color = "#c0392b"
            self.info_color = "#f39c12"
            
            # Configure root background
            self.root.configure(bg=self.bg_color)
            logger.info("Styles configured successfully")
        except Exception as e:
            logger.error(f"Error configuring styles: {str(e)}")
    
    def load_model(self):
        """Load the pre-trained model and scaler with error handling"""
        try:
            model_path = 'models/random_forest_model.pkl'
            scaler_path = 'models/scaler.pkl'
            
            # Check if model files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            # Load model
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
            logger.info("✓ Model and scaler loaded successfully")
            
        except FileNotFoundError as e:
            self.model_loaded = False
            logger.warning(f"⚠ {str(e)}")
            messagebox.showwarning(
                "Model Not Found",
                f"Model files not found.\n\n{str(e)}\n\nPlease run train_model.py first."
            )
        except Exception as e:
            self.model_loaded = False
            logger.error(f"Error loading model: {str(e)}")
            messagebox.showerror(
                "Model Load Error",
                f"Failed to load model: {str(e)}"
            )
    
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
        try:
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
        except Exception as e:
            logger.error(f"Error creating header: {str(e)}")
            raise
    
    def create_input_frame(self):
        """Create input fields for medical data with validation"""
        try:
            input_frame = ttk.Frame(self.root, padding="20")
            input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Title
            input_title = ttk.Label(
                input_frame,
                text="Medical Information",
                font=("Helvetica", 14, "bold")
            )
            input_title.grid(row=0, column=0, columnspan=4, pady=10)
            
            # Input fields configuration with descriptions
            self.fields = {
                'Pregnancies': {'row': 1, 'col': 0, 'range': (0, 17), 'unit': '', 
                               'description': 'Number of times pregnant'},
                'Glucose': {'row': 1, 'col': 1, 'range': (0, 200), 'unit': 'mg/dL',
                           'description': 'Plasma glucose concentration'},
                'BloodPressure': {'row': 1, 'col': 2, 'range': (0, 122), 'unit': 'mm Hg',
                                 'description': 'Diastolic blood pressure'},
                'SkinThickness': {'row': 1, 'col': 3, 'range': (0, 99), 'unit': 'mm',
                                 'description': 'Triceps skin fold thickness'},
                'Insulin': {'row': 2, 'col': 0, 'range': (0, 846), 'unit': 'mu U/ml',
                           'description': '2-Hour serum insulin'},
                'BMI': {'row': 2, 'col': 1, 'range': (0, 67.1), 'unit': 'kg/m²',
                       'description': 'Body mass index'},
                'DiabetesPedigreeFunction': {'row': 2, 'col': 2, 'range': (0, 2.42), 'unit': '',
                                            'description': 'Diabetes pedigree function'},
                'Age': {'row': 2, 'col': 3, 'range': (21, 81), 'unit': 'years',
                       'description': 'Age in years'}
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
            
            # Note about ranges and example
            note_frame = ttk.Frame(input_frame)
            note_frame.grid(row=4, column=0, columnspan=4, pady=10)
            
            note_label = ttk.Label(
                note_frame,
                text="💡 Example: Pregnancies=1, Glucose=120, BP=80, SkinThickness=20, Insulin=30, BMI=25, DPF=0.3, Age=30",
                font=("Helvetica", 9, "italic"),
                foreground="gray"
            )
            note_label.pack()
            
            logger.info("Input frame created successfully")
        except Exception as e:
            logger.error(f"Error creating input frame: {str(e)}")
            raise
    
    def create_button_frame(self):
        """Create prediction and reset buttons"""
        try:
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
            
            logger.info("Button frame created successfully")
        except Exception as e:
            logger.error(f"Error creating button frame: {str(e)}")
            raise
    
    def create_result_frame(self):
        """Create frame for displaying prediction results"""
        try:
            result_frame = ttk.LabelFrame(
                self.root,
                text="Prediction Result",
                padding="20"
            )
            result_frame.pack(fill=tk.BOTH, padx=10, pady=10)
            
            # Result text
            self.result_text = scrolledtext.ScrolledText(
                result_frame,
                height=8,
                width=80,
                font=("Courier", 11),
                bg="#f9f9f9",
                relief=tk.FLAT,
                state=tk.DISABLED
            )
            self.result_text.pack(fill=tk.BOTH, expand=True)
            
            logger.info("Result frame created successfully")
        except Exception as e:
            logger.error(f"Error creating result frame: {str(e)}")
            raise
    
    def validate_input(self, user_data):
        """Validate user input with comprehensive error handling"""
        errors = []
        warnings = []
        
        try:
            # Check for empty fields
            for field_name, value in user_data.items():
                if value is None or str(value).strip() == '':
                    errors.append(f"• {field_name} is empty")
            
            if errors:
                return False, errors, warnings
            
            # Validate numeric values
            for field_name, value in list(user_data.items()):
                try:
                    user_data[field_name] = float(value)
                except ValueError:
                    errors.append(f"• {field_name} must be a number (got '{value}')")
            
            if errors:
                return False, errors, warnings
            
            # Validate ranges
            for field_name, value in user_data.items():
                min_val, max_val = self.fields[field_name]['range']
                if not (min_val <= value <= max_val):
                    warnings.append(
                        f"• {field_name}: {value} is outside typical range ({min_val}-{max_val})"
                    )
            
            return True, errors, warnings
            
        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors, warnings
    
    def predict(self):
        """Make prediction based on user input with comprehensive error handling"""
        if not self.model_loaded:
            messagebox.showerror(
                "Error",
                "Model not loaded. Please ensure model files exist and run train_model.py first."
            )
            logger.error("Prediction attempted without loaded model")
            return
        
        try:
            # Get user inputs
            user_data = {}
            for field_name, entry in self.entries.items():
                value = entry.get().strip()
                user_data[field_name] = value if value else None
            
            # Validate input
            is_valid, errors, warnings = self.validate_input(user_data)
            
            if not is_valid:
                error_msg = "Please fix the following errors:\n\n" + "\n".join(errors)
                messagebox.showerror("Input Validation Error", error_msg)
                logger.warning(f"Input validation failed: {errors}")
                return
            
            # Show warnings if any
            if warnings:
                warning_msg = "Warning - Some values are outside typical ranges:\n\n" + "\n".join(warnings)
                warning_msg += "\n\nDo you want to continue?"
                if messagebox.askyesno("Input Warning", warning_msg) == False:
                    logger.info("Prediction cancelled by user due to warnings")
                    return
            
            # Create DataFrame for prediction
            df = pd.DataFrame([user_data])
            
            # Check for NaN or infinite values
            if df.isnull().any().any():
                raise ValueError("Invalid numerical values detected")
            
            # Scale features
            try:
                df_scaled = self.scaler.transform(df)
            except Exception as e:
                logger.error(f"Error scaling features: {str(e)}")
                raise ValueError(f"Failed to process features: {str(e)}")
            
            # Make prediction
            try:
                prediction = self.model.predict(df_scaled)[0]
                probability = self.model.predict_proba(df_scaled)[0][1]
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                raise ValueError(f"Model prediction failed: {str(e)}")
            
            # Display results
            self.display_results(prediction, probability, user_data)
            logger.info(f"Prediction successful: {prediction} (probability: {probability:.4f})")
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror(
                "Prediction Error",
                f"An error occurred during prediction:\n\n{str(e)}"
            )
    
    def display_results(self, prediction, probability, user_data):
        """Display prediction results in the result frame"""
        try:
            # Determine diagnosis
            if prediction == 0:
                diagnosis = "NO DIABETES DETECTED ✓"
                risk_level = "LOW RISK"
                confidence = (1 - probability) * 100
            else:
                diagnosis = "DIABETES DETECTED ⚠"
                risk_level = "HIGH RISK"
                confidence = probability * 100
            
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
            
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")
            messagebox.showerror("Display Error", f"Failed to display results: {str(e)}")
    
    def reset_fields(self):
        """Clear all input fields"""
        try:
            for entry in self.entries.values():
                entry.delete(0, tk.END)
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            
            logger.info("All fields cleared")
        except Exception as e:
            logger.error(f"Error resetting fields: {str(e)}")
            messagebox.showerror("Reset Error", f"Failed to clear fields: {str(e)}")

def main():
    """Main function to run the GUI application"""
    try:
        root = tk.Tk()
        app = DiabetesRiskPredictorGUI(root)
        logger.info("Application started successfully")
        root.mainloop()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}\n{traceback.format_exc()}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()
