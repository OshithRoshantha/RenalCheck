import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('model/bestModel.pkl')
scaler = joblib.load('model/scaler.pkl')

features = {
    "Serum phosphate level": {"type": "number", "min": 0.5, "max": 6.0, "step": 0.1, "default": 3.5},
    "Random blood glucose level (mg/dl)": {"type": "number", "min": 50, "max": 500, "step": 1, "default": 100},
    "Duration of diabetes mellitus (years)": {"type": "number", "min": 0, "max": 50, "step": 1, "default": 5},
    "C-reactive protein (CRP) level": {"type": "number", "min": 0, "max": 10, "step": 0.1, "default": 0.5},
    "Blood pressure (mm/Hg)": {"type": "number", "min": 70, "max": 200, "step": 1, "default": 120},
    "Cystatin C level": {"type": "number", "min": 0.5, "max": 2.5, "step": 0.1, "default": 0.8},
    "Age of the patient": {"type": "number", "min": 18, "max": 100, "step": 1, "default": 50},
    "Sodium level (mEq/L)": {"type": "number", "min": 120, "max": 160, "step": 0.1, "default": 140},
    "Body Mass Index (BMI)": {"type": "number", "min": 15, "max": 50, "step": 0.1, "default": 25},
    "Red blood cell count (millions/cumm)": {"type": "number", "min": 3, "max": 6, "step": 0.1, "default": 4.5},
    "Urine protein-to-creatinine ratio": {"type": "number", "min": 0, "max": 5, "step": 0.1, "default": 0.2},
    "Cholesterol level": {"type": "number", "min": 100, "max": 300, "step": 1, "default": 180},
    "Urine output (ml/day)": {"type": "number", "min": 500, "max": 3000, "step": 100, "default": 1500},
    "Appetite (good/poor)": {"type": "categorical", "options": ["Good", "Poor"]},
    "Hypertension (yes/no)": {"type": "categorical", "options": ["No", "Yes"]},
    "Sugar in urine": {"type": "categorical", "options": ["No", "Yes"]},
    "Pedal edema (yes/no)": {"type": "categorical", "options": ["No", "Yes"]},
    "Albumin in urine": {"type": "categorical", "options": ["No", "Yes"]},
    "Pus cells in urine": {"type": "categorical", "options": ["No", "Yes"]},
    "Bacteria in urine": {"type": "categorical", "options": ["No", "Yes"]},
    "Anemia (yes/no)": {"type": "categorical", "options": ["No", "Yes"]},
    "Red blood cells in urine": {"type": "categorical", "options": ["No", "Yes"]},
    "Smoking status": {"type": "categorical", "options": ["No", "Yes"]},
    "Diabetes mellitus (yes/no)": {"type": "categorical", "options": ["No", "Yes"]},
    "Urinary sediment microscopy results": {"type": "categorical", "options": ["Normal", "Abnormal"]},
    "Pus cell clumps in urine": {"type": "categorical", "options": ["No", "Yes"]}
}

def main():
    st.title("Kidney Disease Risk Prediction")
    
    with st.form("patient_form"):
        inputs = {}
        cols = st.columns(2)
        
        for i, (feature, config) in enumerate(features.items()):
            with cols[i % 2]:
                if config["type"] == "number":
                    inputs[feature] = st.number_input(
                        label=feature,
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=config["step"]
                    )
                else:
                    inputs[feature] = st.radio(
                        label=feature,
                        options=config["options"]
                    )
        
        submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        try:
            input_df = pd.DataFrame([inputs])
            
            for feature, config in features.items():
                if config["type"] == "categorical":
                    for option in config["options"]:
                        col_name = f"{feature}_{option}" if len(config["options"]) > 2 else f"{feature}_{option}"
                        input_df[col_name] = (inputs[feature] == option).astype(int)
            
            expected_columns = [
                "Serum phosphate level", "Random blood glucose level (mg/dl)",
                "Duration of diabetes mellitus (years)", "C-reactive protein (CRP) level",
                "Blood pressure (mm/Hg)", "Cystatin C level", "Age of the patient",
                "Sodium level (mEq/L)", "Body Mass Index (BMI)", 
                "Red blood cell count (millions/cumm)", "Urine protein-to-creatinine ratio",
                "Cholesterol level", "Urine output (ml/day)",
                "Appetite (good/poor)_Good", "Appetite (good/poor)_Poor",
                "Hypertension (yes/no)_No", "Hypertension (yes/no)_Yes",
                "Sugar in urine_No", "Sugar in urine_Yes",
                "Pedal edema (yes/no)_No", "Pedal edema (yes/no)_Yes",
                "Albumin in urine_No", "Albumin in urine_Yes",
                "Pus cells in urine_No", "Pus cells in urine_Yes",
                "Bacteria in urine_No", "Bacteria in urine_Yes",
                "Anemia (yes/no)_No", "Anemia (yes/no)_Yes",
                "Red blood cells in urine_No", "Red blood cells in urine_Yes",
                "Smoking status_No", "Smoking status_Yes",
                "Diabetes mellitus (yes/no)_No", "Diabetes mellitus (yes/no)_Yes",
                "Urinary sediment microscopy results_Normal", "Urinary sediment microscopy results_Abnormal",
                "Pus cell clumps in urine_No", "Pus cell clumps in urine_Yes"
            ]
            
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[expected_columns]
            processed_input = scaler.transform(input_df)
            prediction = model.predict(processed_input)
            proba = model.predict_proba(processed_input)
            
            risk_levels = {
                0: "No Disease", 
                1: "Low Risk",
                2: "Moderate Risk",
                3: "Severe Disease",
                4: "High Risk"
            }
            
            st.subheader("Prediction Results")
            st.metric("Risk Level", f"{prediction[0]} - {risk_levels[prediction[0]]}")
            
            st.write("Probability Distribution:")
            proba_df = pd.DataFrame({
                "Risk Level": risk_levels.values(),
                "Probability": proba[0]
            })
            st.bar_chart(proba_df.set_index("Risk Level"))
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()