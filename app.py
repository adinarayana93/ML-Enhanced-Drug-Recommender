import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import tempfile

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="MediScan - Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Model Loader (with correct paths)
# -----------------------------
@st.cache_resource
def load_model_files():
    model_path = os.path.join("saved_models", "hybrid_model.pkl")
    vect_path = os.path.join("saved_models", "vectorizer.pkl")
    enc_path = os.path.join("saved_models", "encoder.pkl")
    data_path = os.path.join("data", "merged_df_cleaned.csv")

    if not os.path.exists(model_path):
        st.error("❌ Missing model file: hybrid_model.pkl in saved_models folder.")
        st.stop()
    if not os.path.exists(data_path):
        st.error("❌ Missing dataset file: merged_df_cleaned.csv in data folder.")
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    encoder = joblib.load(enc_path)
    data = pd.read_csv(data_path)
    return model, vectorizer, encoder, data


model, vectorizer, encoder, data = load_model_files()

# -----------------------------
# Storage Path (cross-platform safe)
# -----------------------------
def get_data_path():
    base_dir = ".streamlit" if os.environ.get('STREAMLIT_SHARING') else tempfile.gettempdir()
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "patient_records.csv")


PATIENT_DATA_FILE = get_data_path()

# -----------------------------
# Save Patient Data
# -----------------------------
def save_patient_data(name, age, symptoms, disease, meds, diet, workout, precautions):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        "Timestamp": timestamp,
        "Patient Name": name,
        "Patient Age": age,
        "Symptoms": ", ".join(symptoms),
        "Predicted Disease": disease,
        "Medications": meds,
        "Diet Recommendations": diet,
        "Workout Recommendations": workout,
        "Precautions": precautions
    }])

    if os.path.exists(PATIENT_DATA_FILE):
        old = pd.read_csv(PATIENT_DATA_FILE)
        df = pd.concat([old, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(PATIENT_DATA_FILE, index=False)
    return df

# -----------------------------
# Get Symptom Options
# -----------------------------
symptom_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
symptom_options = sorted(data[symptom_cols].stack().dropna().unique())

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e88e5, #1565c0);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
}
.main-title { font-size: 2.5rem; font-weight: 700; }
.subtitle { font-size: 1.2rem; opacity: 0.9; }
.card {
    background-color: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.section-title {
    color: #1e88e5;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 2px solid #1e88e5;
}
.precaution-item {
    background-color: #f0f4f8;
    border-left: 3px solid #ff5722;
    padding: 0.6rem;
    border-radius: 5px;
    margin-bottom: 0.5rem;
}
.result-card {
    background-color: #fff;
    border-radius: 10px;
    padding: 2rem;
    border-top: 5px solid #4caf50;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.9rem;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">MediScan</h1>
    <p class="subtitle">AI-Driven Disease Prediction & Health Recommendations</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🩺 Disease Prediction", "📁 Patient Records"])

with tab1:
    col1, col2 = st.columns([1, 3])

    # Left panel
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
        name = st.text_input("Full Name", placeholder="Enter name")
        age = st.number_input("Age", 0, 120, 25)
        st.markdown('<div class="section-title">🔍 Symptom Analysis</div>', unsafe_allow_html=True)
        st.write("Select symptoms:")

        selected_symptoms = []
        for i in range(4):
            s = st.selectbox(f"Symptom {i+1}", ["None"] + list(symptom_options), key=f"sym{i}")
            if s != "None":
                selected_symptoms.append(s)

        examine = st.button("🔎 Examine Symptoms", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Right panel
    with col2:
        if examine:
            if not name:
                st.warning("Please enter patient name.")
            elif not selected_symptoms:
                st.warning("Select at least one symptom.")
            else:
                with st.spinner("Analyzing..."):
                    combined = " ".join(selected_symptoms)
                    pred_code = model.predict(vectorizer.transform([combined]))[0]
                    disease = encoder.inverse_transform([pred_code])[0]
                    info = data[data["Disease"] == disease].iloc[0]

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader(f"🩺 Predicted Disease: {disease}")
                st.write(f"**Patient:** {name} ({age} yrs)")
                st.write(f"**Symptoms:** {', '.join(selected_symptoms)}")

                st.markdown('<div class="section-title">💊 Treatment</div>', unsafe_allow_html=True)
                meds = info.get("Medication", "Not Available")
                st.write(meds)

                st.markdown('<div class="section-title">🥗 Lifestyle</div>', unsafe_allow_html=True)
                diet = info.get("Diet", "Not Available")
                workout = info.get("workout", "Not Available")
                st.write(f"**Diet:** {diet}")
                st.write(f"**Workout:** {workout}")

                st.markdown('<div class="section-title">⚕️ Precautions</div>', unsafe_allow_html=True)
                precautions = []
                for i in range(1, 5):
                    p = info.get(f"Precaution_{i}")
                    if isinstance(p, str) and p.strip():
                        precautions.append(p)
                        st.markdown(f'<div class="precaution-item">✓ {p}</div>', unsafe_allow_html=True)

                precautions_text = ", ".join(precautions) if precautions else "No precautions available."
                save_patient_data(name, age, selected_symptoms, disease, meds, diet, workout, precautions_text)
                st.success("Record saved successfully!")
                st.info("⚠️ Note: This tool is for educational and research use only.")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("👋 Enter patient info and select symptoms to begin analysis.")

# -----------------------------
# Patient Records Tab
# -----------------------------
with tab2:
    st.markdown('<div class="section-title">📋 Patient Records</div>', unsafe_allow_html=True)
    if os.path.exists(PATIENT_DATA_FILE):
        df = pd.read_csv(PATIENT_DATA_FILE)
        if len(df) > 0:
            search = st.text_input("Search by patient name or disease:")
            if search:
                df = df[df["Patient Name"].str.contains(search, case=False) | df["Predicted Disease"].str.contains(search, case=False)]
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "patient_records.csv", "text/csv")
        else:
            st.info("No records yet. Run a prediction first.")
    else:
        st.info("No patient records found.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(f"""
<div class="footer">
<p>MediScan Disease Prediction System © {datetime.now().year}</p>
<p>Developed by Adinarayana Pantham</p>
</div>
""", unsafe_allow_html=True)
