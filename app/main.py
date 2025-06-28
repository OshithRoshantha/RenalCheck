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
    "C-reactive protein (CRP) level": {"type": "number", "min": 0.0, "max": 10.0, "step": 0.1, "default": 0.5},
    "Blood pressure (mm/Hg)": {"type": "number", "min": 70, "max": 200, "step": 1, "default": 120},
    "Cystatin C level": {"type": "number", "min": 0.5, "max": 2.5, "step": 0.1, "default": 0.8},
    "Age of the patient": {"type": "number", "min": 18, "max": 100, "step": 1, "default": 50},
    "Sodium level (mEq/L)": {"type": "number", "min": 120.0, "max": 160.0, "step": 0.1, "default": 140.0},
    "Body Mass Index (BMI)": {"type": "number", "min": 15.0, "max": 50.0, "step": 0.1, "default": 25.0},
    "Red blood cell count (millions/cumm)": {"type": "number", "min": 3.0, "max": 6.0, "step": 0.1, "default": 4.5},
    "Urine protein-to-creatinine ratio": {"type": "number", "min": 0.0, "max": 5.0, "step": 0.1, "default": 0.2},
    "Cholesterol level": {"type": "number", "min": 100, "max": 300, "step": 1, "default": 180},
    "Urine output (ml/day)": {"type": "number", "min": 500, "max": 3000, "step": 100, "default": 1500},
    
    "Appetite (good/poor)": {"type": "categorical", "options": ["good", "poor"]},
    "Hypertension (yes/no)": {"type": "categorical", "options": ["no", "yes"]},
    "Sugar in urine": {"type": "categorical", "options": ["no", "yes"]},
    "Pedal edema (yes/no)": {"type": "categorical", "options": ["no", "yes"]},
    "Albumin in urine": {"type": "categorical", "options": ["no", "yes"]},
    "Pus cells in urine": {"type": "categorical", "options": ["no", "yes"]},
    "Bacteria in urine": {"type": "categorical", "options": ["no", "yes"]},
    "Anemia (yes/no)": {"type": "categorical", "options": ["no", "yes"]},
    "Red blood cells in urine": {"type": "categorical", "options": ["no", "yes"]},
    "Smoking status": {"type": "categorical", "options": ["no", "yes"]},
    "Diabetes mellitus (yes/no)": {"type": "categorical", "options": ["no", "yes"]},
    "Urinary sediment microscopy results": {"type": "categorical", "options": ["normal", "abnormal"]},
    "Pus cell clumps in urine": {"type": "categorical", "options": ["no", "yes"]}
}

def main():
    st.title("Kidney Disease Risk Detection 🤖")
    
    with st.form("patient_form"):
        inputs = {}
        cols = st.columns(2)
        
        for i, (feature, config) in enumerate(features.items()):
            with cols[i % 2]:
                if config["type"] == "number":
                    if feature in ["Duration of diabetes mellitus (years)", "Age of the patient", 
                                 "Blood pressure (mm/Hg)", "Urine output (ml/day)"]:
                        inputs[feature] = st.number_input(
                            feature,
                            min_value=int(config["min"]),
                            max_value=int(config["max"]),
                            value=int(config["default"]),
                            step=int(config["step"])
                        )
                    else:
                        inputs[feature] = st.number_input(
                            feature,
                            min_value=float(config["min"]),
                            max_value=float(config["max"]),
                            value=float(config["default"]),
                            step=float(config["step"])
                        )
                else:

                    display_options = [opt.capitalize() for opt in config["options"]]
                    selected = st.radio(
                        feature,
                        options=display_options,
                        index=0
                    )
                    inputs[feature] = selected.lower()
        
        submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        try:
            input_df = pd.DataFrame([inputs])[list(features.keys())]
            for feature, config in features.items():
                if config["type"] == "categorical":
                    for option in config["options"]:
                        col_name = f"{feature}_{option}"
                        input_df[col_name] = (input_df[feature] == option).astype(int)
            
            input_df = input_df.drop(columns=[f for f in features if features[f]["type"] == "categorical"])
            
            expected_features = model.get_booster().feature_names
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[expected_features]
            
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