# data_explore_and_preprocess.py
"""
Day 2:
- Load merged_df.csv
- Inspect columns, types, missing values
- Create Combined_Symptoms (join Symptom_1..Symptom_4)
- Save cleaned CSV to data/merged_df_cleaned.csv
- Print a short summary for verification
"""


import os
import pandas as pd


DATA_DIR = "data"
INPUT_FN = os.path.join(DATA_DIR, "merged_df.csv")
OUTPUT_FN = os.path.join(DATA_DIR, "merged_df_cleaned.csv")

def safe_read_csv(path):
    """Read CSV and return DataFrame or raise a helpful error."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}\nPut merged_df.csv into the data/ folder.")
    
    return pd.read_csv(path)


def main():
    # 1) Load dataset

    df = safe_read_csv(INPUT_FN)
    print("Loaded dataframe. Shape:", df.shape)
    print("\nColumn list:\n", df.columns.tolist())

    # 2) Basic inspection
    print("\nSample rows (first 5):")
    print(df.head(5).to_string(index=False))

    print("\nDtypes and non-null counts:")
    print(df.info())

    # 3) Missing values summary
    print("\nMissing values per column:")
    print(df.isna().sum())

    # 4) Ensure symptom columns exists (create if missing)
    symptom_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
    for c in symptom_cols:
        if c not in df.columns:
            print(f"Warning: '{c}' not found in dataset. Creating empty column.")
            df[c] = ""

    # 5) Create Combined_Symptoms (join non_empty symptom strings)
    def combine_symptoms(row):
        #collect non-empty and non-null symptom values, strip spaces
        vals = []
        for c in symptom_cols:
            v = row.get(c, "")
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s != "" and s.lower() not in ("nan", "none", "0"):
                vals.append(s)
            
        return " ".join(vals)
    
    df["Combined_Symptoms"] = df.apply(combine_symptoms, axis=1)

    # 6) Basic target (Diseases) checks
    if "Disease" not in df.columns:
        raise KeyError("Excepted column 'Disease' not found in merged_df.csv")
    print("\nNumber of unique diseases:", df["Disease"].nunique())
    print("\nTop 10 most frequent diseases:")
    print(df["Disease"].value_counts().head(10))


    # 7) Preview combined symptoms
    print("\nPreview Combined_Symptoms (first 8 rows):")
    for i, s in enumerate(df["Combined_Symptoms"].head(8), start=1):
        print(f"{i:2d}. {s[:200]}")

    
    # 8) Save cleaned CSV (small & safe copy)
    os.makedirs(DATA_DIR, exist_ok = True)
    df.to_csv(OUTPUT_FN, index = False)
    print(f"\nSaved cleaned dataframes to: {OUTPUT_FN}")


    # 9) Quick sanity checks: small sample counts
    print("\nSanity checks:")
    #any empty Combined_Symptoms?
    empty_count = (df["Combined_Symptoms"].str.strip() == "").sum()
    print("Rows with empty Combined_Symptoms:", int(empty_count))
    # class distribution summary
    class_counts = df["Disease"].value_counts()
    print("Total rows:", len(df), "Total classes:", class_counts.shape[0])
    print("5 smallest classes by support (may be rare classes):")
    print(class_counts.tail(5))

if __name__ == "__main__":
    main()
