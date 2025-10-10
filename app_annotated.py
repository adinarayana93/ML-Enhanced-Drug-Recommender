# app_annotated.py
"""
Annotated Streamlit app for the ML-Enhanced Drug Recommender.
This version is heavily commented so you can explain every step to your guide.
It expects the following files to exist in the repo:
 - data/merged_df_cleaned.csv        (cleaned dataset)
 - saved_models/hybrid_model.pkl     (trained VotingClassifier)
 - saved_models/vectorizer.pkl       (CountVectorizer fitted on Combined_Symptoms)
 - saved_models/encoder.pkl          (LabelEncoder fitted on Disease)
"""

import streamlit as st
import joblib
import pandas as pd
import requests
import os
from datetime import datetime
import numpy as np

# --- Paths ---
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
DATA_CLEAN = os.path.join("data", "merged_df_cleaned.csv")
PATIENT_RECORDS = "patient_records.csv"

# --- GitHub model URLs (replace with yours if needed) ---
RELEASE_ARTIFACTS = {
    "hybrid_model.pkl": "https://github.com/adinarayana93/ML-Enhanced-Drug-Recommender/releases/download/v1.0/hybrid_model.pkl",
    "vectorizer.pkl": "https://github.com/adinarayana93/ML-Enhanced-Drug-Recommender/releases/download/v1.0/vectorizer.pkl",
    "encoder.pkl": "https://github.com/adinarayana93/ML-Enhanced-Drug-Recommender/releases/download/v1.0/encoder.pkl"
}

# --- Ensure models exist (downloads if missing) ---
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

# --- Load all artifacts ---
def load_artifacts():
    ensure_models_present()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    encoder = joblib.load(ENC_PATH)
    merged_df = pd.read_csv(DATA_CLEAN)
    return model, vectorizer, encoder, merged_df

model, vectorizer, encoder, merged_df = load_artifacts()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="ML-Enhanced Drug Recommender", layout="wide")
st.title("🧠 ML-Enhanced Drug Recommender — Annotated Version")

st.markdown(
    """
    **How to use:**  
    Enter or select symptoms and press *Predict Disease*.  
    The app uses a trained ensemble model to predict the most likely disease  
    and shows recommended medications, diet, workout and precautions.
    """
)

tab1, tab2 = st.tabs(["Disease Prediction", "Patient Records"])

# --- Helper Functions ---
def combine_symptom_inputs(symptom_inputs):
    tokens = [s.strip() for s in symptom_inputs if isinstance(s, str) and s.strip() != ""]
    return " ".join(tokens)

def get_top_tokens_for_input(input_text, vectorizer, model, encoder, top_n=10):
    """Approximate token-level importance for interpretability."""
    vec = vectorizer.transform([input_text])
    features = vectorizer.get_feature_names_out()
    
    fi_sum = np.zeros(len(features), dtype=float)
    fi_count = 0
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                fi = getattr(est, "feature_importances_")  # ✅ fixed typo (was feature_importance_)
                if len(fi) == len(fi_sum):
                    fi_sum += fi
                    fi_count += 1

    if fi_count == 0:
        return []

    fi_avg = fi_sum / fi_count
    nz = vec.nonzero()[1]
    token_scores = []
    for idx in nz:
        token = features[idx]
        count = vec[0, idx]
        score = float(count) * float(fi_avg[idx])
        token_scores.append((token, score, int(count)))

    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores[:top_n]

def lookup_recommendations(predicted_label, df):
    row = df[df["Disease"] == predicted_label]
    if row.empty:
        return None
    
    row0 = row.iloc[0]
    meds = row0.get("Medication", "")
    diet = row0.get("Diet", "")
    workout = row0.get("workout", "")
    precautions = [row0.get(f"Precaution_{i}", "") for i in range(1, 5) if f"Precaution_{i}" in row0.index]
    precautions = [p for p in precautions if str(p).strip() != ""]
    return {
        "medication": meds,
        "diet": diet,
        "workout": workout,
        "precautions": precautions
    }

def save_patient_record(record: dict, filename=PATIENT_RECORDS):
    df = pd.DataFrame([record])
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# --- Tab 1: Prediction ---
with tab1:
    st.subheader("Patient Information & Symptoms")

    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Patient Name", "")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female", "Other"])
    with col3:
        contact = st.text_input("Contact (optional)", "")

    st.markdown("### Symptoms (choose or type)")
    s1 = st.text_input("Symptom 1", "")
    s2 = st.text_input("Symptom 2", "")
    s3 = st.text_input("Symptom 3", "")
    s4 = st.text_input("Symptom 4", "")

    combined_input = combine_symptom_inputs([s1, s2, s3, s4])
    st.write("**Combined Symptoms Preview:**", combined_input[:300] if combined_input else "(empty)")

    if st.button("Predict Disease"):
        if not combined_input:
            st.warning("Please enter at least one symptom before predicting.")
        else:
            X_input_vec = vectorizer.transform([combined_input])
            pred_code = model.predict(X_input_vec)[0]
            predicted_disease = encoder.inverse_transform([pred_code])[0]

            st.success(f"Predicted Disease: **{predicted_disease}**")

            rec = lookup_recommendations(predicted_disease, merged_df)
            if rec:
                st.markdown("#### 💊 Recommended Medication")
                st.write(rec.get("medication", "Not available"))

                st.markdown("#### 🥗 Diet")
                st.write(rec.get("diet", "Not available"))

                st.markdown("#### 🏋️ Workout")
                st.write(rec.get("workout", "Not available"))

                st.markdown("#### ⚕️ Precautions")
                for p in rec.get("precautions", []):
                    st.write("- " + str(p))
            else:
                st.info("No recommendations found for this disease.")

            token_info = get_top_tokens_for_input(combined_input, vectorizer, model, encoder, top_n=10)
            if token_info:
                st.markdown("#### 🔍 Top Contributing Tokens (approx.)")
                for token, score, count in token_info:
                    st.write(f"- **{token}** (count={count}) — importance score: {score:.5f}")
            else:
                st.info("Token-level explainability not available for this model.")

            record = {
                "datetime": datetime.now().isoformat(),
                "name": name.strip() or "Anonymous",
                "age": int(age),
                "sex": sex,
                "contact": contact,
                "combined_symptoms": combined_input,
                "predicted_disease": predicted_disease
            }
            save_patient_record(record)
            st.info("✅ Patient record saved locally.")

# --- Tab 2: Patient Records ---
with tab2:
    st.subheader("Patient Records")
    if os.path.exists(PATIENT_RECORDS):
        pr_df = pd.read_csv(PATIENT_RECORDS)
        st.dataframe(pr_df.tail(50))

        # CSV download
        csv = pr_df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV", data=csv, file_name="patient_records.csv", mime="text/csv")

        # Excel download
        try:
            import io
            excel_buffer = io.BytesIO()
            pr_df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_bytes = excel_buffer.getvalue()
            st.download_button("📘 Download Excel (.xlsx)", data=excel_bytes,
                               file_name="patient_records.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.write("Excel export not available:", e)
    else:
        st.info("No patient records found yet (records saved locally to patient_records.csv).")

st.markdown("---")
st.markdown("**Note:** This annotated app uses saved models and vectorizer. The model files are downloaded from GitHub if missing.")
