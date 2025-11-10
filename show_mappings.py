import joblib

vec = joblib.load("saved_models/vectorizer.pkl")
enc = joblib.load("saved_models/encoder.pkl")


print("Number of classes:", len(enc.classes_))
print("First 20 classes:", enc.classes_[:20])

try:
    feat = vec.get_feature_names_out()
except:
    feat = list(vec.vocabulary_.keys())

print("Vocab size:", len(feat))
print("First 20 tokens:", feat[:20])