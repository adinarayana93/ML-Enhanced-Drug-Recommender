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
import plotly.express as px


def admin_login():
    st.sidebar.header("🔐 Admin Login")
    username = st.sidebar.text_input("Username", placeholder="Enter username")
    password = st.sidebar.text_input("Password", placeholder="Enter password", type="password")
    if username == "admin" and password == "1234":
        st.sidebar.success("Login successful ✅")
        return True
    elif username or password:
        st.sidebar.error("Invalid credentials ❌")
        return False
    return False

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

is_admin = admin_login()
tab1, tab2, tab3, tab4 = st.tabs(["💊 Disease Prediction", "📁 Patient Records", "📊 Analytics Dashboard", "📝 Feedback"])



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


def generate_disease_summary_pdf(disease_name, df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"Disease Summary Report: {disease_name}", ln=True, align="C")

    subset = df[df["Predicted_Disease"].str.lower() == disease_name.lower()]
    if subset.empty:
        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, "No records found for this disease.", ln=True)
    else:
        pdf.set_font("Arial", "I", 12)
        pdf.cell(200, 10, f"Total Cases: {len(subset)}", ln=True)
        pdf.ln(10)
        for i, row in subset.iterrows():
            pdf.cell(200, 10, f"{row['Date']} — {row['Patient_Name']} ({row['Age']} yrs)", ln=True)

    os.makedirs("reports", exist_ok=True)
    fname = f"reports/{disease_name.replace(' ', '_')}_summary.pdf"
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
# Tab 2: Patient Records (with admin login & summary PDF)
# ─────────────────────────────────────────────
with tab2:
    st.subheader("🧾 Patient History")

    # Only admins can view full records
    if not is_admin:
        st.warning("Please log in as admin from the sidebar to view patient records.")
    else:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)

            # --- Filter Section ---
            st.markdown("### 🔍 Filter Records")
            search_name = st.text_input("Search by Patient Name")
            search_disease = st.text_input("Search by Disease")

            filtered_df = df.copy()
            if search_name:
                filtered_df = filtered_df[
                    filtered_df["Patient_Name"].str.contains(search_name, case=False, na=False)
                ]
            if search_disease:
                filtered_df = filtered_df[
                    filtered_df["Predicted_Disease"].str.contains(search_disease, case=False, na=False)
                ]

            st.dataframe(filtered_df)

            # --- Download Buttons ---
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download Filtered CSV",
                data=csv,
                file_name="patient_records.csv",
                mime="text/csv",
            )

            # Excel download
            try:
                import io
                excel_buf = io.BytesIO()
                filtered_df.to_excel(excel_buf, index=False, engine="openpyxl")
                st.download_button(
                    "📘 Download Excel (.xlsx)",
                    data=excel_buf.getvalue(),
                    file_name="patient_records.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.write("Excel export not available:", e)

            # --- Disease-wise Summary PDF ---
            st.markdown("### 📄 Generate Disease Summary Report")
            disease_names = sorted(df["Predicted_Disease"].dropna().unique().tolist())
            if disease_names:
                selected_disease = st.selectbox("Select Disease", disease_names)
                if st.button("Generate Summary PDF"):
                    summary_path = generate_disease_summary_pdf(selected_disease, df)
                    with open(summary_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Summary PDF",
                            f,
                            file_name=os.path.basename(summary_path),
                            mime="application/pdf",
                        )
            else:
                st.info("No disease records available for summary report.")
        else:
            st.info("No patient records found yet (records saved locally to patient_history.csv).")



# ─────────────────────────────────────────────
# Tab 3: Analytics Dashboard
# ─────────────────────────────────────────────
with tab3:
    if not is_admin:
        st.warning("Please log in as admin from the sidebar to view records.")
    else:
        st.subheader("📊 Analytics Dashboard")

        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            if df.empty:
                st.info("No records yet to display analytics.")
            else:
                st.markdown("### 🔹 Disease Prediction Frequency")
                disease_count = df["Predicted_Disease"].value_counts().reset_index()
                disease_count.columns = ["Disease", "Count"]
                fig1 = px.bar(
                    disease_count,
                    x="Disease",
                    y="Count",
                    color="Count",
                    title="Most Commonly Predicted Diseases",
                    color_continuous_scale="tealgrn"
                )
                fig1.update_layout(xaxis_title="Disease", yaxis_title="Predictions")
                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("### 🔹 Patient Age Distribution")
                if "Age" in df.columns:
                    fig2 = px.histogram(
                        df,
                        x="Age",
                        nbins=10,
                        color_discrete_sequence=["#FFD65C"],
                        title="Patient Age Distribution"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### 🔹 Symptom Frequency (Top 20)")
                all_symptoms = []
                for s in df["Symptoms"].dropna():
                    all_symptoms.extend([sym.strip() for sym in s.split(",")])
                symptom_df = pd.DataFrame(all_symptoms, columns=["Symptom"])
                top_symptoms = symptom_df["Symptom"].value_counts().reset_index().head(20)
                top_symptoms.columns = ["Symptom", "Count"]
                fig3 = px.bar(
                    top_symptoms,
                    x="Symptom",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Bluered_r",
                    title="Most Commonly Reported Symptoms"
                )
                fig3.update_layout(xaxis_title="Symptom", yaxis_title="Frequency")
                st.plotly_chart(fig3, use_container_width=True)

                st.markdown("### 🔹 Records Over Time")
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    time_series = df.groupby(df["Date"].dt.date).size().reset_index(name="Count")
                    fig4 = px.line(
                        time_series,
                        x="Date",
                        y="Count",
                        markers=True,
                        title="Predictions Over Time"
                    )
                    st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No patient history found yet.")


# ─────────────────────────────────────────────
# Tab 4: Feedback
# ─────────────────────────────────────────────
with tab4:
    st.subheader("📝 Share Your Feedback")

    name = st.text_input("Your Name")
    rating = st.slider("How satisfied are you with the system?", 1, 5, 4)
    comment = st.text_area("Your Comments", placeholder="Type your feedback here...")
    feedback_file = os.path.join("data", "feedback.csv")

    if st.button("Submit Feedback"):
        if name and comment:
            record = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": name,
                "Rating": rating,
                "Comment": comment
            }
            os.makedirs("data", exist_ok=True)
            df = pd.DataFrame([record])
            if os.path.exists(feedback_file):
                df.to_csv(feedback_file, mode="a", header=False, index=False)
            else:
                df.to_csv(feedback_file, index=False)
            st.success("Thank you for your feedback! 💬")
        else:
            st.warning("Please fill out all fields before submitting.")

    if os.path.exists(feedback_file) and is_admin:
        st.markdown("### View Collected Feedback")
        fb_df = pd.read_csv(feedback_file)
        st.dataframe(fb_df)




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
