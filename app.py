import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.conformal import run_conformal_prediction  
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
}

st.set_page_config(page_title="Conformal Health Risk Predictor", layout="wide")

st.title("🫀 Conformal Health Risk Predictor")
st.markdown("This app predicts **heart disease risk** using ML + Conformal Prediction for confidence intervals.")

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))

st.sidebar.write("### About")
st.sidebar.info(
    "✅ Trained on UCI Heart Disease dataset\n"
    "✅ Provides **risk probability** + **95% conformal interval**\n"
    "✅ Supports multiple ML models"
)

# -------------------------------
# Example Patients
# -------------------------------
example_patients = {
    "Healthy Patient": [45, 1, 0, 120, 200, 0, 0, 150, 0, 2.5, 1, 0, 1],
    "High Risk Patient": [65, 0, 3, 160, 300, 1, 2, 100, 1, 5.0, 2, 3, 3],
}

if st.sidebar.button("Load Healthy Patient"):
    st.session_state["inputs"] = example_patients["Healthy Patient"]

if st.sidebar.button("Load High Risk Patient"):
    st.session_state["inputs"] = example_patients["High Risk Patient"]

# -------------------------------
# Input Form
# -------------------------------
st.subheader("🔍 Enter Patient Details")

features = [
    "Age of the person",
    "Sex",
    "Chest Pain Type",
    "Resting Blood Pressure (mm Hg)",
    "Serum Cholesterol (mg/dl)",
    "Fasting Blood Sugar > 120 mg/dl",
    "Resting ECG Results",
    "Maximum Heart Rate Achieved",
    "Exercise Induced Angina",
    "ST Depression Induced by Exercise",
    "Slope of Peak Exercise ST Segment",
    "Number of Major Vessels (0–3)",
    "Thalassemia"
]

# Default values
if "inputs" not in st.session_state:
    st.session_state["inputs"] = [50, 1, 0, 130, 250, 0, 0, 150, 0, 2.5, 1, 0, 1]

user_inputs = []

# -------------------------------
# Input fields with full names & dropdowns
# -------------------------------
age = st.number_input("Age of the person", min_value=1, max_value=120, value=st.session_state["inputs"][0], step=1)
user_inputs.append(age)

sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)],
                   index=0 if st.session_state["inputs"][1] == 1 else 1)[1]
user_inputs.append(sex)

cp = st.selectbox("Chest Pain Type", [
    ("Typical Angina", 0),
    ("Atypical Angina", 1),
    ("Non-anginal Pain", 2),
    ("Asymptomatic", 3)
], index=st.session_state["inputs"][2])[1]
user_inputs.append(cp)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250,
                           value=st.session_state["inputs"][3], step=1)
user_inputs.append(trestbps)

chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600,
                       value=st.session_state["inputs"][4], step=1)
user_inputs.append(chol)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("True", 1), ("False", 0)],
                   index=0 if st.session_state["inputs"][5] == 1 else 1)[1]
user_inputs.append(fbs)

restecg = st.selectbox("Resting ECG Results", [
    ("Normal", 0),
    ("ST-T Wave Abnormality", 1),
    ("Left Ventricular Hypertrophy", 2)
], index=st.session_state["inputs"][6])[1]
user_inputs.append(restecg)

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250,
                          value=st.session_state["inputs"][7], step=1)
user_inputs.append(thalach)

exang = st.selectbox("Exercise Induced Angina", [("Yes", 1), ("No", 0)],
                     index=0 if st.session_state["inputs"][8] == 1 else 1)[1]
user_inputs.append(exang)

oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0,
                          value=float(st.session_state["inputs"][9]), step=0.1, format="%.1f")
user_inputs.append(oldpeak)

slope = st.selectbox("Slope of Peak Exercise ST Segment", [
    ("Upsloping", 0),
    ("Flat", 1),
    ("Downsloping", 2)
], index=st.session_state["inputs"][10])[1]
user_inputs.append(slope)

ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3,
                     value=st.session_state["inputs"][11], step=1)
user_inputs.append(ca)

thal = st.selectbox("Thalassemia", [
    ("Normal", 1),
    ("Fixed Defect", 2),
    ("Reversible Defect", 3)
], index=st.session_state["inputs"][12]-1)[1]
user_inputs.append(thal)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk"):
    model = models[model_choice]
    X = np.array(user_inputs).reshape(1, -1)

    # Model probability
    prob = model.predict_proba(X)[0][1] * 100

    # Conformal prediction interval (corrected call)
    lower, upper = run_conformal_prediction(model, X)

    # Risk category
    if prob < 30:
        risk_color, risk_label = "green", "Low Risk"
    elif prob < 60:
        risk_color, risk_label = "orange", "Moderate Risk"
    else:
        risk_color, risk_label = "red", "High Risk"

    # Show result
    st.markdown(
        f"### 🧾 Prediction: <span style='color:{risk_color};font-weight:bold'>{risk_label}</span>",
        unsafe_allow_html=True
    )
    st.write(f"**Risk Probability:** {prob:.2f}%")
    st.write(f"**95% Confidence Interval:** {np.mean(lower):.2f}% – {np.mean(upper):.2f}%")

    # -------------------------------
    # Visualization (Bar Chart)
    # -------------------------------
    st.subheader("📊 Patient Profile vs Normal Range")
    normal_ranges = {
        "Age of the person": (20, 60),
        "Resting Blood Pressure (mm Hg)": (80, 130),
        "Serum Cholesterol (mg/dl)": (150, 240),
        "Maximum Heart Rate Achieved": (100, 180),
        "ST Depression Induced by Exercise": (0, 4)
    }

    fig, ax = plt.subplots()
    for feat, (low, high) in normal_ranges.items():
        val = user_inputs[features.index(feat)]
        ax.bar(feat, val, color="red" if val < low or val > high else "green")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------------
    # Download Report
    # -------------------------------
    report_df = pd.DataFrame([user_inputs], columns=features)
    report_df["Predicted Risk"] = prob
    report_df["Risk Category"] = risk_label
    report_df["Confidence Interval"] = f"{lower:.2f}–{upper:.2f}"

    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Patient Report (CSV)", csv, "patient_report.csv", "text/csv")
