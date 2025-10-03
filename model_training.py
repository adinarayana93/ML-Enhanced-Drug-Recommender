# model_training.py
"""
Day 3:
Train hybrid ML model for disease prediction
- Load cleaned dataset (data/merged_df_cleaned.csv)
- Vectorize Combined_Symptoms
- Encode Disease
- Train ensemble model (RF + GB + XGB) using VotingClassifier
- Evaluate metrics
- Save model, vectorizer, encoder
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATA_FN = os.path.join("data", "merged_df_cleaned.csv")
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok = True)

def main():
    df = pd.read_csv(DATA_FN)
    print("Loaded dataset:", df.shape)

    X = df["Combined_Symptoms"]
    y = df["Disease"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Classes (sample):", list(label_encoder.classes_)[:10], "...")
    print("Total unique diseases:", len(label_encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size = 0.2, random_state = 42, stratify = y_encoded
    )
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

    rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)

    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")

    hybrid_model = VotingClassifier(
        estimators=[("rf", rf_model), ("gb", gb_model), ("xgb", xgb_model)],
        voting="hard"
    )

    print("\nTraining hybrid model...")
    hybrid_model.fit(X_train_vec, y_train)

    y_pred = hybrid_model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average= "weighted", zero_division=0)

    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")

    joblib.dump(hybrid_model, os.path.join(MODEL_DIR, "hybrid_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    print("\nSaved model, vectorizer, encoder in:", MODEL_DIR)


if __name__ == "__main__":
    main()