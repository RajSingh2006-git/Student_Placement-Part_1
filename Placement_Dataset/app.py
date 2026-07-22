import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Placement Predictor")
st.write("Enter the student's IQ and CGPA to predict placement status.")

# File paths
MODEL_PATH = '/Users/rajsingh/Desktop/Machine_learning/Placement_Dataset/model.pkl'
SCALER_PATH = '/Users/rajsingh/Desktop/Machine_learning/Placement_Dataset/scaler.pkl'
DATASET_PATH = '/Users/rajsingh/Desktop/Machine_learning/Placement_Dataset/placement_dataset (1).csv'

if not os.path.exists(MODEL_PATH) and os.path.exists('model.pkl'):
    MODEL_PATH = 'model.pkl'
if not os.path.exists(SCALER_PATH) and os.path.exists('scaler.pkl'):
    SCALER_PATH = 'scaler.pkl'
if not os.path.exists(DATASET_PATH) and os.path.exists('placement_dataset (1).csv'):
    DATASET_PATH = 'placement_dataset (1).csv'

@st.cache_resource
def load_artifacts():
    model, scaler = None, None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    elif os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH).drop_duplicates()
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(df[['iq', 'cgpa']])
        
    return model, scaler

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("Error: model.pkl not found!")
    st.stop()

# Inputs
col1, col2 = st.columns(2)
with col1:
    iq = st.number_input("IQ (Intelligence Quotient)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
with col2:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

if st.button("Predict Placement", type="primary", use_container_width=True):
    scaled_input = scaler.transform(np.array([[iq, cgpa]]))
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100
    
    st.markdown("---")
    if prediction == 1:
        st.balloons()
        st.success(f"🥳 **Result: CONGRATULATIONS! PLACED!** (Placement Chance: {probability:.1f}%)")
        st.markdown("### 🎉 Congratulations on Your Success!")
        st.markdown("Your hard work has paid off! You are well-positioned for placement. Keep shining and keep growing! ⭐🚀")
    else:
        st.info(f"💡 **Result: NOT PLACED YET** (Placement Chance: {probability:.1f}%)")
        st.markdown("### 😊 Keep Your Head Up & Stay Positive!")
        st.markdown("""
        Every setback is a setup for an even greater comeback! 🌟
        
        **Here's how you can boost your chances:**
        - 📚 **Focus on CGPA**: Work consistently on your core academic modules.
        - 💡 **Sharpen Problem Solving**: Practice daily coding & aptitude challenges.
        - 🚀 **Build Projects**: Enhance your resume with hands-on practical projects.
        
        *Keep improving your skills, stay confident, and try again! Your dream opportunity is right around the corner!* 💪✨
        """)
