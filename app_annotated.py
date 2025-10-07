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
import os
from datetime import datetime


MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
DATA_CLEAN = os.path.join("data", "merged_df_cleaned.csv")
PATIENT_RECORDS = "patient_records.csv"


def load_artifacts():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    encoder = joblib.load(ENC_PATH)

    merged_df = pd.read_csv(DATA_CLEAN)
    return model, vectorizer, encoder, merged_df

model, vectorizer, encoder, merged_df = load_artifacts()


st.set_page_config(page_title = "ML-Enhanced Drug Recommender", layout = "wide")
st.title("ML-Enhanced Drug Recommender -- Annotated")

st.markdown(
    """
    **How to use:** Enter or select symptoms, press *Predict Disease*. The app uses a trained ensemble
    model to predict the most likely disease and shows recommended medications, diet, workout and precautions.
    """
)

tab1, tab2 = st.tabs(["Disease Prediction", "Patient Records"])


def combine_symptom_inputs(symptom_inputs):
    tokens = [s.strip() for s in symptom_inputs if isinstance(s, str) and s.strip() != ""]
    return " ".join(tokens)


def get_top_tokens_for_input(input_text, vectorizer, model, encoder, top_n=10):
    """
    Returns top_n tokens (word, score) that contributed for this input_text.
    Approach:
      - vectorize input -> counts for tokens present
      - get feature importances from tree-based estimators inside VotingClassifier
      - combine importances across estimators that have feature_importances_
      - compute token_score = token_count * feature_importance
      - return top tokens sorted by token_score
    Note: This is a simple, intuitive approximation (not SHAP), but useful for demonstration.
    """ 

    vec = vectorizer.transform([input_text])
    try:
        features = vectorizer.get_feature_names_out()
    except:
        features = list(vectorizer.vocabulary_.keys())
    
    import numpy as np
    fi_sum = np.zeros(len(features), dtype=float)
    fi_count = 0
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                fi = getattr(est, "feature_importance_")
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
        df.to_csv(filename, mode = "a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    


with tab1:
    st.subheader("Patient information & symptoms")

    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Patient Name", value = "")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female", "Other"])
    with col3:
        contact = st.text_input("Contact (optional)", value = "")

    st.markdown("### Symptoms (choose or type)")
    s1 = st.text_input("Symptom 1", "")
    s2 = st.text_input("Symptom 2", "")
    s3 = st.text_input("Symptom 3", "")
    s4 = st.text_input("Symptom 4", "")


    combined_input = combine_symptom_inputs([s1, s2, s3, s4])
    st.write("Combined Symptoms preview:", combined_input[:300] if combined_input else "(empty)")


    if st.button("Predict Disease"):
        if not combined_input:
            st.warning("Please enter at least one symptom before predicting.")
        else:
            X_input_vec = vectorizer.transform([combined_input])
            pred_code = model.predict(X_input_vec)[0]
            predicted_disease = encoder.inverse_transform([pred_code])[0]

            st.success(f"Predicted disease: **{predicted_disease}**")

            rec = lookup_recommendations(predicted_disease, merged_df)
            if rec:
                st.markdown("#### Recommended Medication")
                st.write(rec.get("medication", "Not available"))

                st.markdown("#### Diet")
                st.write(rec.get("diet", "Not available"))

                st.markdown("#### Workout")
                st.write(rec.get("workout", "Not available"))

                st.markdown("#### Precautions")
                for p in rec.get("precautions", []):
                    st.write("- " + str(p))

            else:
                st.info("No recommendation found in the database for this disease.")

                        # --- Explainability: show top tokens ---
            token_info = get_top_tokens_for_input(combined_input, vectorizer, model, encoder, top_n=10)
            if token_info:
                st.markdown("#### Top contributing tokens (approx.)")
                # token_info is list of (token, score, count)
                for token, score, count in token_info:
                    st.write(f"- **{token}** (count={count}) — importance score: {score:.5f}")
            else:
                st.info("Token-level explainability not available for this model.")

            if len(name.strip()) == 0:
                name_to_save = "Anonymous"
            else:
                name_to_save = name.strip()

            record = {
                "datetime": datetime.now().isoformat(),
                "name": name_to_save,
                "age": int(age),
                "sex": sex,
                "contact": contact,
                "combined_symptoms": combined_input,
                "predicted_disease": predicted_disease
            }
            save_patient_record(record)
            st.info("Patient record saved locally.")

        
with tab2:
    st.subheader("Patient Records")
    if os.path.exists(PATIENT_RECORDS):
        pr_df = pd.read_csv(PATIENT_RECORDS)
        st.dataframe(pr_df.tail(50))

                # Provide CSV download
        csv = pr_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="patient_records.csv", mime="text/csv")

        # Provide Excel download (xlsx)
        try:
            import io
            excel_buffer = io.BytesIO()
            pr_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_bytes = excel_buffer.getvalue()
            st.download_button("Download Excel (.xlsx)", data=excel_bytes, file_name="patient_records.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.write("Excel export not available:", e)

    else:
        st.info("No patient records found yet (records saved locally to patient_records.csv).")

st.markdown("---")
st.markdown("**Note:** This annotated app uses saved models and vectorizer. The model files are not committed to GitHub; they are large binary artifacts.")
