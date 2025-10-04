# inspect_artifacts.py
"""
Day 4:
Inspect trained artifacts for the ML-Enhanced Drug Recommender
- Load encoder, vectorizer, model
- Print classes, vocab size, and shapes
- Run a sample prediction
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FN = "data/merged_df_cleaned.csv"
MODEL_DIR = "saved_models"

def main():
    df = pd.read_csv(DATA_FN)

    if "Combined_Symptoms" not in df.columns:
        raise KeyError("Combined_Symptoms not found. Run Day2 preprocessing first.")
    
    encoder = joblib.load(f"{MODEL_DIR}/encoder.pkl")
    vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
    model = joblib.load(f"{MODEL_DIR}/hybrid_model.pkl")

    print("\n=== Artifact Inspection ===")
    print("Total diseases:", len(encoder.classes_))
    print("Sample disease classes:", encoder.classes_[:10], "...")


    try:
        vocab = vectorizer.get_feature_names_out()
    except:
        vocab = list(vectorizer.vocabulary_.keys())
    print("Vocabulary size:", len(vocab))
    print("Sample vocab tokens:", vocab[:15])

    X = df["Combined_Symptoms"]
    y = encoder.transform(df["Disease"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )    
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\nData shapes:")
    print("X_train_vec:", X_train_vec.shape)
    print("X_test_vec:", X_test_vec.shape)

    sample_text = X.iloc[0:1]
    sample_vec = vectorizer.transform(sample_text)
    pred_code = model.predict(sample_vec)[0]
    pred_label = encoder.inverse_transform([pred_code])[0]

    print("\nSample input text:", sample_text.iloc[0])
    print("Predicted disease:", pred_label)

if __name__ == "__main__":
    main()



