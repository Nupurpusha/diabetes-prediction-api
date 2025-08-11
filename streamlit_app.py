import streamlit as st
import requests

# --- CONFIG ---
API_URL = "http://127.0.0.1:8000/predict_diabetes/"  # Local FastAPI endpoint

# --- PAGE SETUP ---
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# --- HEADER ---
st.markdown(
    """
    <h1 style='text-align: center; color: orange;'>Diabetes Prediction</h1>
    <p style='text-align: center; font-size:18px;'>
    Predict the probability of having Diabetes
    </p>
    """,
    unsafe_allow_html=True
)

# --- FORM LAYOUT ---
with st.form(key="diabetes_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        skin_thickness = st.number_input("SkinThickness", min_value=0)
        diabetes_pedigree_function = st.number_input("DiabetesPedigreeFunction", min_value=0.0, format="%.3f")

    with col2:
        glucose = st.number_input("Glucose", min_value=0)
        insulin = st.number_input("Insulin", min_value=0)
        age = st.number_input("Age", min_value=0, step=1)

    with col3:
        blood_pressure = st.number_input("BloodPressure", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f")

    submit_btn = st.form_submit_button("PREDICT PROBABILITY")

# --- BACKEND CALL ---
if submit_btn:
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
            st.info(f"Probability: **{result['probability']:.2%}**")
        else:
            st.error(f"Error from API: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
