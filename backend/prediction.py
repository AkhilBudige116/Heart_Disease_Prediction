"""
prediction.py
--------------
Loads the already-trained model (models/heart_disease_model.pkl) and
exposes a simple, reusable prediction function. The model is NOT
retrained here -- it is only loaded from disk, exactly as required.
"""

import os
import joblib
import pandas as pd

from backend.preprocessing import FEATURE_COLUMNS

# Path to the trained model file (relative to project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'heart_disease_model.pkl')


def load_trained_model(model_path: str = MODEL_PATH):
    """Load the trained model from disk using joblib."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python train_model.py' first to train and save the model."
        )
    model = joblib.load(model_path)
    return model


def predict_heart_disease(age, cholesterol, max_heart_rate, blood_pressure, old_peak, model=None):
    """
    Run a single prediction using the trained model.

    Parameters correspond exactly to the notebook's feature order:
        Age, Cholesterol, Max_Heart_Rate, Blood_Pressure, Old_Peak

    Returns a dict with:
        - 'prediction': raw 0/1 model output
        - 'result_text': human-readable result string
    """
    if model is None:
        model = load_trained_model()

    # Build a DataFrame with the exact feature names and order the model
    # was trained on -- this avoids sklearn's "feature names mismatch"
    # warnings/errors and keeps prediction logic explicit and readable.
    input_df = pd.DataFrame(
        [[age, cholesterol, max_heart_rate, blood_pressure, old_peak]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result_text = "Heart Disease Detected"
    else:
        result_text = "No Heart Disease Detected"

    return {
        'prediction': int(prediction),
        'result_text': result_text,
    }
