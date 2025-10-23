# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt  # For advanced charts
import io                 # For download button

# --- Page Configuration ---
st.set_page_config(
    page_title="RiskBound AI",  # CHANGED
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Main Page Layout ---

st.title("🫀 RiskBound AI") # CHANGED
st.markdown("This app predicts **heart disease risk** using a machine learning model wrapped with **Conformal Prediction** to provide trustworthy, guaranteed confidence sets.")

# --- Load Preprocessors and Calibrated Models ---
# (This logic is efficient and unchanged)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model(path):
    """Loads a single model or preprocessor."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {path}. Please run setup scripts.")
        return None
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

# Load the preprocessors
imputer = load_model(os.path.join(MODEL_DIR, "imputer.pkl"))
scaler = load_model(os.path.join(MODEL_DIR, "scaler.pkl"))

# Load the CALIBRATED MAPIE models
models = {
    "Random Forest": load_model(os.path.join(MODEL_DIR, "mapie_random_forest.pkl")),
    "Logistic Regression": load_model(os.path.join(MODEL_DIR, "mapie_logistic_regression.pkl")),
    "SVM": load_model(os.path.join(MODEL_DIR, "mapie_svm.pkl")),
    "XGBoost": load_model(os.path.join(MODEL_DIR, "mapie_xgboost.pkl")),
}

# Stop the app if core files are missing
if not all([imputer, scaler] + list(models.values())):
    st.error("One or more essential model files are missing. Stopping app.")
    st.stop()


# --- Sidebar ---
# (This logic is clean and unchanged)
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Choose a Model", list(models.keys()))
    alpha = st.slider("Confidence Level (1 - α)", 0.80, 0.99, 0.95, 0.01)
    st.markdown(f"This will compute a **{int(alpha*100)}%** confidence prediction set.")

    st.write("### About")
    st.info(
        "✅ Trained on UCI Heart Disease dataset\n"
        f"✅ Provides **risk probability** + **{int(alpha*100)}% conformal prediction set**\n"
        "✅ Supports multiple ML models"
    )
    
    st.header("Load Example Patients")
    example_patients = {
        "Healthy Patient": [45, 1, 0, 120, 200, 0, 0, 150, 0, 1.0, 1, 0, 2],
        "High Risk Patient": [65, 0, 3, 160, 300, 1, 2, 100, 1, 5.0, 2, 3, 3],
    }

    if st.button("Load Healthy Patient"):
        st.session_state["inputs"] = example_patients["Healthy Patient"]
    if st.button("Load High Risk Patient"):
        st.session_state["inputs"] = example_patients["High Risk Patient"]


# --- Main Page Layout ---

st.title("🫀 Conformal Health Risk Predictor")
st.markdown("This app predicts **heart disease risk** using a machine learning model wrapped with **Conformal Prediction** to provide trustworthy, guaranteed confidence sets.")

# -------------------------------
# Input Form (Now in an Expander)
# -------------------------------
features = [
    "Age of the person", "Sex", "Chest Pain Type", "Resting Blood Pressure (mm Hg)",
    "Serum Cholesterol (mg/dl)", "Fasting Blood Sugar > 120 mg/dl", "Resting ECG Results",
    "Maximum Heart Rate Achieved", "Exercise Induced Angina", "ST Depression Induced by Exercise",
    "Slope of Peak Exercise ST Segment", "Number of Major Vessels (0–3)", "Thalassemia"
]
feature_keys = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", 
    "exang", "oldpeak", "slope", "ca", "thal"
]

if "inputs" not in st.session_state:
    st.session_state["inputs"] = example_patients["High Risk Patient"]

with st.expander("🔍 Click to Enter Patient Details", expanded=True):
    user_inputs = []
    cols = st.columns(3)

    # --- Column 1 ---
    age = cols[0].number_input(features[0], min_value=1, max_value=120, value=st.session_state["inputs"][0], step=1)
    user_inputs.append(age)
    sex = cols[0].selectbox(features[1], [("Male", 1), ("Female", 0)], index=0 if st.session_state["inputs"][1] == 1 else 1)[1]
    user_inputs.append(sex)
    cp = cols[0].selectbox(features[2], [("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)], index=st.session_state["inputs"][2])[1]
    user_inputs.append(cp)
    trestbps = cols[0].number_input(features[3], min_value=50, max_value=250, value=st.session_state["inputs"][3], step=1)
    user_inputs.append(trestbps)

    # --- Column 2 ---
    chol = cols[1].number_input(features[4], min_value=100, max_value=600, value=st.session_state["inputs"][4], step=1)
    user_inputs.append(chol)
    fbs = cols[1].selectbox(features[5], [("True > 120 mg/dl", 1), ("False <= 120 mg/dl", 0)], index=0 if st.session_state["inputs"][5] == 1 else 1)[1]
    user_inputs.append(fbs)
    restecg = cols[1].selectbox(features[6], [("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], index=st.session_state["inputs"][6])[1]
    user_inputs.append(restecg)
    thalach = cols[1].number_input(features[7], min_value=60, max_value=250, value=st.session_state["inputs"][7], step=1)
    user_inputs.append(thalach)

    # --- Column 3 ---
    exang = cols[2].selectbox(features[8], [("Yes", 1), ("No", 0)], index=0 if st.session_state["inputs"][8] == 1 else 1)[1]
    user_inputs.append(exang)
    oldpeak = cols[2].number_input(features[9], min_value=0.0, max_value=10.0, value=float(st.session_state["inputs"][9]), step=0.1, format="%.1f")
    user_inputs.append(oldpeak)
    slope = cols[2].selectbox(features[10], [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], index=st.session_state["inputs"][10])[1]
    user_inputs.append(slope)
    ca = cols[2].number_input(features[11], min_value=0, max_value=4, value=st.session_state["inputs"][11], step=1)
    user_inputs.append(ca)
    thal = cols[2].selectbox(features[12], [("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)], index=st.session_state["inputs"][12]-1 if st.session_state["inputs"][12] > 0 else 0)[1]
    user_inputs.append(thal)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("Predict Risk", type="primary", use_container_width=True):
    # 1. Convert to numpy array
    X_raw = np.array(user_inputs).reshape(1, -1)
    
    # 2. Apply preprocessing (Impute and Scale)
    try:
        X_imputed = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imputed)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # 3. Get the selected calibrated model
    mapie_model = models[model_choice]
    
    # 4. Get simple probability from the base estimator
    prob = mapie_model.estimator.predict_proba(X_scaled)[0][1] * 100

    # 5. Get conformal prediction SET
    y_pred, y_pis = mapie_model.predict(X_scaled, alpha=(1-alpha))
    
    prediction_set_booleans = y_pis[0, :, 0] 
    prediction_set = []
    if prediction_set_booleans[0]: prediction_set.append("Low Risk (0)")
    if prediction_set_booleans[1]: prediction_set.append("High Risk (1)")
    set_string = f"{{{', '.join(prediction_set)}}}" if prediction_set else "{}"
    
    # Risk category (from simple probability)
    if prob < 30: risk_color, risk_label = "green", "Low Risk"
    elif prob < 60: risk_color, risk_label = "orange", "Moderate Risk"
    else: risk_color, risk_label = "red", "High Risk"

    # --- Display Results in a Tabbed Container ---
    st.header("Prediction Results", divider="rainbow")
    with st.container(border=True):
        
        tab1, tab2, tab3 = st.tabs(["📈 Key Metrics", "🩺 Patient Profile Chart", "📄 Download Report"])

        # --- Tab 1: Key Metrics ---
        with tab1:
            st.markdown(
                f"### Top Prediction: <span style='color:{risk_color};font-weight:bold'>{risk_label}</span>",
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            col1.metric("Risk Probability", f"{prob:.2f}%")
            col2.metric(f"{int(alpha*100)}% Confidence Prediction Set", set_string)
            
            st.divider()
            
            # Explain the result
            st.markdown("#### How to Interpret This")
            st.write(f"The model's single best guess is a **{prob:.2f}%** chance of heart disease ({risk_label}).")
            
            if len(prediction_set) > 1:
                st.warning(f"**Uncertain Prediction:** The {int(alpha*100)}% confidence set is **{set_string}**. "
                           f"This means the model is *not* confident enough to distinguish between Low and High risk at this confidence level. "
                           f"The true category is guaranteed to be in this set {int(alpha*100)}% of the time.", icon="⚠️")
            elif len(prediction_set) == 1:
                st.success(f"**Confident Prediction:** The {int(alpha*100)}% confidence set is **{set_string}**. "
                           f"The model is confident in its prediction. The true category is guaranteed to be "
                           f"in this set {int(alpha*100)}% of the time.", icon="✅")
            else: 
                st.error(f"**Empty Prediction Set:** {set_string}. This is a rare event (called 'absurd'), but it means the model "
                         "is so confident the prediction is wrong that it returns nothing. "
                         "This can indicate your input data is very different from the training data.", icon="❌")

        # --- Tab 2: Patient Profile Chart ---
        with tab2:
            st.subheader("Patient Vitals vs. Normal Ranges")
            
            normal_ranges = {
                "Age": (20, 60),
                "Resting BP": (80, 130),
                "Cholesterol": (150, 240),
                "Max Heart Rate": (100, 180),
                "ST Depression": (0, 1) # Normal is < 1.0
            }
            
            plot_data = {
                "Age": user_inputs[feature_keys.index("age")],
                "Resting BP": user_inputs[feature_keys.index("trestbps")],
                "Cholesterol": user_inputs[feature_keys.index("chol")],
                "Max Heart Rate": user_inputs[feature_keys.index("thalach")],
                "ST Depression": user_inputs[feature_keys.index("oldpeak")]
            }

            source_data = []
            for feat, (low, high) in normal_ranges.items():
                val = plot_data[feat]
                status = "Normal" if (val >= low and val <= high) else "Out of Range"
                source_data.append({
                    "Metric": feat,
                    "Patient Value": val,
                    "Normal Range": f"{low}–{high}",
                    "Status": status
                })
            source_df = pd.DataFrame(source_data)

            # Create the Altair chart
            chart = alt.Chart(source_df).mark_bar().encode(
                x=alt.X('Metric:N', title=None),
                y=alt.Y('Patient Value:Q', title='Patient Value'),
                color=alt.Color('Status:N',
                                scale=alt.Scale(domain=['Normal', 'Out of Range'],
                                                range=['#2ca02c', '#d62728']), # Green, Red
                                legend=alt.Legend(title="Status")
                               ),
                tooltip=['Metric', 'Patient Value', 'Normal Range', 'Status']
            ).properties(
                title="Key Metrics vs. Normal Range"
            ).interactive() 

            st.altair_chart(chart, use_container_width=True)
            
            # --- Download Chart Button ---
            html_chart = chart.to_html()
            st.download_button(
                label="Download Chart as HTML",
                data=html_chart,
                file_name="patient_profile_chart.html",
                mime="text/html"
            )
            st.caption("Open the downloaded HTML file in your browser to save as a PDF or image.")

        # --- Tab 3: Download Full Report ---
        with tab3:
            st.subheader("Download Full Patient Report")
            
            report_df = pd.DataFrame([user_inputs], columns=features)
            report_df["Predicted Risk (%)"] = f"{prob:.2f}"
            report_df["Risk Category"] = risk_label
            report_df[f"{int(alpha*100)}% Confidence Set"] = set_string
            report_df["Model Used"] = model_choice

            csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Report (CSV)", 
                data=csv, 
                file_name="patient_report.csv", 
                mime="text/csv",
                use_container_width=True
            )
            st.dataframe(report_df)

# --- Credits Footer ---
st.divider()
st.markdown(
    """
    <p style='text-align: center; color: gray;'>
        Developed by <a href='https://www.linkedin.com/in/sahuishaan22/' target='_blank'>Ishaan Sahu</a>
    </p>
    """,
    unsafe_allow_html=True
)
