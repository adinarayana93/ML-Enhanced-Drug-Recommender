# app_annotated.py
"""
Annotated Streamlit app for the ML-Enhanced Drug Recommender.
This version includes full functionality up to Day 10:
 - Symptom dropdowns
 - Explainability (SHAP + fallback)
 - Patient history saving
 - PDF report generation
 - CSV / Excel download
"""

import streamlit as st
import joblib
import pandas as pd
import requests
import os
from datetime import datetime
import numpy as np
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import io


# ─────────────────────────────────────────────
# Paths & artifacts
# ─────────────────────────────────────────────
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
DATA_CLEAN = os.path.join("data", "merged_df_cleaned.csv")
HISTORY_FILE = os.path.join("data", "patient_history.csv")

RELEASE_ARTIFACTS = {
    "hybrid_model.pkl": "https://github.com/adinarayana93/ML-Enhanced-Drug-Recommender/releases/download/v1.0/hybrid_model.pkl",
    "vectorizer.pkl": "https://github.com/adinarayana93/ML-Enhanced-Drug-Recommender/releases/download/v1.0/vectorizer.pkl",
    "encoder.pkl": "https://github.com/adinarayana93/ML-Enhanced-Drug-Recommender/releases/download/v1.0/encoder.pkl"
}


# ─────────────────────────────────────────────
# Ensure models exist (auto-download)
# ─────────────────────────────────────────────
def ensure_models_present(model_dir="saved_models"):
    os.makedirs(model_dir, exist_ok=True)
    for name, url in RELEASE_ARTIFACTS.items():
        dest = os.path.join(model_dir, name)
        if not os.path.exists(dest):
            print(f"Downloading {name} ...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)
            print("Downloaded", dest)


# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
def load_artifacts():
    ensure_models_present()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    encoder = joblib.load(ENC_PATH)
    merged_df = pd.read_csv(DATA_CLEAN)
    return model, vectorizer, encoder, merged_df


model, vectorizer, encoder, merged_df = load_artifacts()

# ─────────────────────────────────────────────
# Streamlit setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="ML-Enhanced Drug Recommender", layout="wide")

st.markdown("""
<style>
.main { background-color: #111; color: #eee; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3, h4 { color: #FFD65C; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#222;padding:18px;border-radius:8px;border:1px solid #444;text-align:center;'>
  <h1 style='margin:0;color:#FFD65C;'>🧠 ML-Enhanced Drug Recommender</h1>
  <p style='color:#ccc;'>AI-powered diagnosis support & personalized drug recommendations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**How to use:**  
Select symptoms and click **Predict Disease**.  
The app predicts the most likely disease and recommends medication, diet, workout, and precautions.
""")

tab1, tab2 = st.tabs(["💊 Disease Prediction", "📁 Patient Records"])


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def get_all_symptoms(df):
    s = set()
    for _, row in df.iterrows():
        for i in range(1, 5):
            val = row.get(f"Symptom_{i}")
            if isinstance(val, str) and val.strip():
                s.add(val.strip())
    return sorted(list(s))


ALL_SYMPTOMS = get_all_symptoms(merged_df)


def combine_symptom_inputs(s1, s2, s3, s4):
    symptoms = [s for s in [s1, s2, s3, s4] if s and s.strip()]
    return " ".join(symptoms)


def lookup_recommendations(label, df):
    row = df[df["Disease"] == label]
    if row.empty:
        return None
    r = row.iloc[0]
    meds = r.get("Medication", "")
    diet = r.get("Diet", "")
    workout = r.get("workout", "")
    prec = [r.get(f"Precaution_{i}", "") for i in range(1, 5)]
    prec = [p for p in prec if str(p).strip()]
    return {"medication": meds, "diet": diet, "workout": workout, "precautions": prec}


def compute_shap_for_input(input_text, vectorizer, model, encoder, top_n=10):
    x_vec = vectorizer.transform([input_text])
    try:
        features = vectorizer.get_feature_names_out()
    except Exception:
        features = list(vectorizer.vocabulary_.keys())

    nz = x_vec.nonzero()[1].tolist()
    if not nz:
        return []

    shap_sum = np.zeros(len(features))
    shap_count = 0

    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            try:
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(x_vec)
                if isinstance(sv, list):
                    pred_class = est.predict(x_vec)[0]
                    vals = np.array(sv[pred_class])[0]
                else:
                    vals = np.array(sv)[0]
                shap_sum += vals
                shap_count += 1
            except Exception:
                continue

    if shap_count == 0:
        return []

    shap_mean = shap_sum / shap_count
    tokens = [(features[i], abs(shap_mean[i]), shap_mean[i]) for i in nz]
    tokens.sort(key=lambda x: x[1], reverse=True)
    return tokens[:top_n]


def get_top_tokens_for_input(input_text, vectorizer, model, top_n=10):
    vec = vectorizer.transform([input_text])
    features = vectorizer.get_feature_names_out()

    fi_sum = np.zeros(len(features))
    fi_count = 0
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                fi = est.feature_importances_
                if len(fi) == len(fi_sum):
                    fi_sum += fi
                    fi_count += 1

    if fi_count == 0:
        return []

    fi_avg = fi_sum / fi_count
    nz = vec.nonzero()[1]
    scores = [(features[i], fi_avg[i]) for i in nz]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def save_patient_record(name, age, contact, symptoms, disease):
    rec = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient_Name": name,
        "Age": age,
        "Contact": contact,
        "Symptoms": ", ".join(symptoms),
        "Predicted_Disease": disease,
    }
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([rec])
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)


def generate_pdf_report(name, age, contact, symptoms, disease, rec):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "ML-Enhanced Drug Recommender Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Patient: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age}", ln=True)
    pdf.cell(200, 10, f"Contact: {contact}", ln=True)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, "", ln=True)

    pdf.cell(200, 10, "Symptoms:", ln=True)
    pdf.multi_cell(200, 8, ", ".join(symptoms))
    pdf.cell(200, 10, "", ln=True)
    pdf.cell(200, 10, f"Predicted Disease: {disease}", ln=True)
    pdf.cell(200, 10, "", ln=True)

    pdf.cell(200, 10, "Recommendations:", ln=True)
    pdf.set_font("Arial", "I", 11)
    for k, v in rec.items():
        if v:
            if isinstance(v, list):
                v = ", ".join(v)
            pdf.multi_cell(200, 8, f"{k.capitalize()}: {v}")

    os.makedirs("reports", exist_ok=True)
    fname = f"reports/{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(fname)
    return fname


# ─────────────────────────────────────────────
# Tab 1: Prediction
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Enter Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Name", placeholder="Enter patient's full name")
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col3:
        contact = st.text_input("Contact", placeholder="Phone or email")

    st.subheader("Select Symptoms")
    c1, c2 = st.columns(2)
    with c1:
        s1 = st.selectbox("Symptom 1", ALL_SYMPTOMS)
        s2 = st.selectbox("Symptom 2", ALL_SYMPTOMS)
    with c2:
        s3 = st.selectbox("Symptom 3", ALL_SYMPTOMS)
        s4 = st.selectbox("Symptom 4", ALL_SYMPTOMS)

    if st.button("🔍 Predict Disease & Suggest Drugs"):
        symptoms = [s1, s2, s3, s4]
        combined = combine_symptom_inputs(s1, s2, s3, s4)

        pred_code = model.predict(vectorizer.transform([combined]))[0]
        pred_label = encoder.inverse_transform([pred_code])[0]
        st.success(f"Predicted Disease: {pred_label}")

        rec = lookup_recommendations(pred_label, merged_df)
        if rec:
            st.markdown(f"**Medication:** {rec['medication']}")
            st.markdown(f"**Diet:** {rec['diet']}")
            st.markdown(f"**Workout:** {rec['workout']}")
            st.markdown(f"**Precautions:** {', '.join(rec['precautions'])}")

        if name.strip():
            save_patient_record(name, age, contact, symptoms, pred_label)

        st.markdown("#### Model Explainability")
        shap_tokens = compute_shap_for_input(combined, vectorizer, model, encoder, top_n=10)

        if shap_tokens:
            toks = [t for t, _, _ in shap_tokens]
            vals = [v for _, _, v in shap_tokens]
            colors = ["green" if v >= 0 else "red" for v in vals]
            fig, ax = plt.subplots(figsize=(6, 3))
            y = np.arange(len(toks))
            ax.barh(y, vals, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(toks)
            ax.invert_yaxis()
            ax.set_xlabel("SHAP Value")
            st.pyplot(fig)
        else:
            st.warning("SHAP not available — showing token importance.")
            toks = [t for t, _ in get_top_tokens_for_input(combined, vectorizer, model, top_n=10)]
            vals = [v for _, v in get_top_tokens_for_input(combined, vectorizer, model, top_n=10)]
            fig, ax = plt.subplots(figsize=(6, 3))
            y = np.arange(len(toks))
            ax.barh(y, vals, color="skyblue")
            ax.set_yticks(y)
            ax.set_yticklabels(toks)
            ax.invert_yaxis()
            ax.set_xlabel("Token Importance")
            st.pyplot(fig)

        if rec and st.button("📄 Generate PDF Report"):
            file_path = generate_pdf_report(name, age, contact, symptoms, pred_label, rec)
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download Report", f, file_name=os.path.basename(file_path))


# ─────────────────────────────────────────────
# Tab 2: Patient History
# ─────────────────────────────────────────────
with tab2:
    st.subheader("🧾 Patient History")
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV", data=csv, file_name="patient_records.csv", mime="text/csv")

        excel_buf = io.BytesIO()
        df.to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button("📘 Download Excel", data=excel_buf.getvalue(),
                           file_name="patient_records.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("No patient history found yet.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧬 How it Works")
st.markdown("""
1. User selects up to 4 symptoms.
2. ML model predicts the most likely disease.
3. System recommends medications, diet, workout, and precautions.
4. SHAP shows which symptoms influenced prediction.
5. User can download a PDF report or view past records.
""")
st.markdown("---")
st.markdown(f"""
**Records:** {len(merged_df)} |  
**Symptoms:** {len(ALL_SYMPTOMS)} |  
**Diseases:** {len(encoder.classes_)}
""")
