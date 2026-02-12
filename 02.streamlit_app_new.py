import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np

# --- 1. Page Config ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# --- 2. Helper Functions (Model Loading) ---
def resolve_model_path(filename):
    possible_paths = [
        os.path.join("models", filename),
        os.path.join("src", filename),
        filename,
        os.path.join("..", "models", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def load_model():
    model_filename = "best_model.pkl"
    model_path = resolve_model_path(model_filename)
    if model_path:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# --- 3. Main App ---
def main():
    st.title("🫀 Heart Disease Prediction App (Kaggle S6E2)")
    
    model = load_model()
    if model is None:
        st.error("🚨 Model file (`best_model.pkl`) not found!")
        return

    tab1, tab2 = st.tabs(["📝 Manual Prediction", "📁 Batch Prediction (CSV)"])

    # --- TAB 1: MANUAL INPUT ---
    with tab1:
        st.subheader("Enter Patient Details")
        with st.form("patient_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", 10, 100, 50)
                
                # Sex: 1=Male, 0=Female
                sex_disp = st.selectbox("Sex", ["Male", "Female"])
                sex_val = 1 if sex_disp == "Male" else 0
                
                # Chest Pain: 1=Typical, 2=Atypical, 3=Non-Anginal, 4=Asymptomatic
                cp_disp = st.selectbox("Chest Pain Type", 
                                       ["Typical Angina (1)", "Atypical Angina (2)", "Non-Anginal Pain (3)", "Asymptomatic (4)"])
                cp_map = {"Typical Angina (1)": 1, "Atypical Angina (2)": 2, 
                          "Non-Anginal Pain (3)": 3, "Asymptomatic (4)": 4}
                cp_val = cp_map[cp_disp]
                
                trestbps = st.number_input("Resting Blood Pressure (BP)", 90, 200, 120)

            with col2:
                chol = st.number_input("Cholesterol", 100, 600, 200)
                
                # FBS: 1=True, 0=False
                fbs_disp = st.selectbox("Fasting BS > 120 mg/dl", ["False (0)", "True (1)"])
                fbs_val = 1 if "True" in fbs_disp else 0
                
                # ECG: 0=Normal, 1=ST, 2=LVH
                ecg_disp = st.selectbox("Resting ECG Results", ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
                ecg_map = {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2}
                ecg_val = ecg_map[ecg_disp]

                thalach = st.number_input("Max Heart Rate", 60, 220, 150)

            with col3:
                # ExAng: 1=Yes, 0=No
                exang_disp = st.selectbox("Exercise Angina", ["No (0)", "Yes (1)"])
                exang_val = 1 if "Yes" in exang_disp else 0
                
                oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 0.0, step=0.1)
                
                # Slope: 1=Up, 2=Flat, 3=Down
                slope_disp = st.selectbox("Slope of ST", ["Upsloping (1)", "Flat (2)", "Downsloping (3)"])
                slope_map = {"Upsloping (1)": 1, "Flat (2)": 2, "Downsloping (3)": 3}
                slope_val = slope_map[slope_disp]
                
                ca = st.selectbox("Number of vessels fluro (0-3)", [0, 1, 2, 3])
                
                # Thallium: 3=Normal, 6=Fixed, 7=Reversible
                thal_disp = st.selectbox("Thallium Test", ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"])
                thal_map = {"Normal (3)": 3, "Fixed Defect (6)": 6, "Reversible Defect (7)": 7}
                thal_val = thal_map[thal_disp]

            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Construct input data with correct numerical mapping
                input_data = {
                    'Age': [age],
                    'Sex': [sex_val],                
                    'Chest pain type': [cp_val],     
                    'BP': [trestbps],
                    'Cholesterol': [chol],
                    'FBS over 120': [fbs_val],       
                    'EKG results': [ecg_val],        
                    'Max HR': [thalach],
                    'Exercise angina': [exang_val],  
                    'ST depression': [oldpeak],      
                    'Slope of ST': [slope_val],      
                    'Number of vessels fluro': [ca], 
                    'Thallium': [thal_val]           
                }
                
                input_df = pd.DataFrame(input_data)
                
                # Optional: Show debug table
                # st.write("Model Input (Processed):")
                # st.dataframe(input_df)
                
                try:
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0.0
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Risk Score", f"%{proba*100:.1f}")
                    
                    if prediction == 1 or prediction == "Presence":
                        c2.error("🚨 PREDICTION: HEART DISEASE DETECTED")
                    else:
                        c2.success("✅ PREDICTION: NORMAL")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    # --- TAB 2: CSV UPLOAD ---
    with tab2:
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(df.head())
                
                if st.button("Start Batch Analysis"):
                    try:
                        preds = model.predict(df)
                        df['Prediction'] = preds
                        
                        if hasattr(model, "predict_proba"):
                             df['Probability'] = model.predict_proba(df)[:, 1]

                        st.success("Analysis Completed!")
                        st.dataframe(df.head())
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Ensure CSV columns match the training data format (numeric/int).")
                        
            except Exception as e:
                st.error("Could not read file.")

if __name__ == "__main__":
    main()