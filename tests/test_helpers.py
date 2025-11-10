# tests/test_helpers.py
import pandas as pd
import os
import tempfile
import pytest

# Import helper functions from app_annotated (adjust if you kept helpers in different file)
from app_annotated import combine_symptom_inputs, lookup_recommendations

def test_combine_symptom_inputs_basic():
    inputs = ["fever", " cough ", "", None]
    combined = combine_symptom_inputs(inputs)
    assert "fever" in combined
    assert "cough" in combined
    assert combined == "fever cough" or combined == "fever  cough".replace("  ", " ")

def test_lookup_recommendations_found(tmp_path):
    # Create a tiny dataframe and write to a temp csv, then test lookup_recommendations uses it
    df = pd.DataFrame([
        {"Disease": "TestDisease", "Medication": "MedA", "Diet": "DietA", "workout": "Walk", "Precaution_1": "P1"},
    ])
    # Pass df directly to function and ensure we get expected values
    rec = lookup_recommendations("TestDisease", df)
    assert rec is not None
    assert rec["medication"] == "MedA"
    assert rec["diet"] == "DietA"
    assert "P1" in rec["precautions"]

def test_lookup_recommendations_not_found():
    df = pd.DataFrame([{"Disease": "Other", "Medication": "X"}])
    rec = lookup_recommendations("Missing", df)
    assert rec is None
