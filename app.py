"""
app.py
------
Streamlit frontend for the Heart Disease Prediction application.

This app loads the already-trained model (does NOT retrain on every
prediction) and lets the user enter patient information to get a
prediction. It also shows a simple model performance comparison table.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from backend.preprocessing import load_cleaned_data, get_features_and_target
from backend.model import compare_models, get_candidate_models, train_model
from backend.prediction import predict_heart_disease
from backend.ai_advisor import get_ai_precautions

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'heart_disease_cleaned.csv')
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("Heart Disease Prediction")
st.write(
    "This app predicts whether a patient is likely to have heart disease "
    "based on Age, Cholesterol, Maximum Heart Rate, Blood Pressure, and Old Peak, "
    "using the machine-learning algorithm selected in the sidebar."
)

# ---------------------------------------------------------------------------
# Algorithm selection and model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model(algorithm_name):
    """Train and cache the selected algorithm using the cleaned dataset."""
    data = load_cleaned_data(DATA_PATH)
    features, target = get_features_and_target(data)
    model = get_candidate_models()[algorithm_name]
    return train_model(model, features, target)


with st.sidebar:
    st.header("Prediction Algorithm")
    selected_algorithm = st.selectbox(
        "Choose a model",
        options=["Logistic Regression", "KNN", "SVC", "XGBoost"],
        index=2,
        help="The selected model is trained on the cleaned heart-disease dataset and used for your prediction.",
    )
    st.caption(f"Active model: **{selected_algorithm}**")

with st.spinner(f"Loading {selected_algorithm}..."):
    model = get_model(selected_algorithm)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
    cholesterol = st.number_input("Cholesterol", min_value=0.0, value=200.0, step=1.0)
    max_heart_rate = st.number_input("Maximum Heart Rate", min_value=0.0, value=150.0, step=1.0)

with col2:
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=120.0, step=1.0)
    old_peak = st.number_input("Old Peak", min_value=0.0, value=1.0, step=0.1, format="%.1f")

predict_clicked = st.button("Predict Heart Disease", type="primary")

if predict_clicked:
    # ---- Basic validation ----
    errors = []
    if age <= 0 or age > 120:
        errors.append("Please enter a realistic Age (1-120).")
    if cholesterol < 0:
        errors.append("Cholesterol cannot be negative.")
    if max_heart_rate < 0:
        errors.append("Maximum Heart Rate cannot be negative.")
    if blood_pressure < 0:
        errors.append("Blood Pressure cannot be negative.")
    if old_peak < 0:
        errors.append("Old Peak cannot be negative.")

    if errors:
        for err in errors:
            st.warning(err)
    else:
        result = predict_heart_disease(
            age=age,
            cholesterol=cholesterol,
            max_heart_rate=max_heart_rate,
            blood_pressure=blood_pressure,
            old_peak=old_peak,
            model=model,
        )

        if result['prediction'] == 1:
            st.error(f"Result: {result['result_text']}")
        else:
            st.success(f"Result: {result['result_text']}")

        st.subheader("AI Precautions and Next Steps")
        if not os.environ.get("GROQ_API_KEY"):
            st.info("Add your Groq API key to the `.env` file to receive AI guidance.")
        else:
            with st.spinner("Generating personalized precautions..."):
                try:
                    guidance = get_ai_precautions(
                        result_text=result['result_text'],
                        age=age,
                        cholesterol=cholesterol,
                        max_heart_rate=max_heart_rate,
                        blood_pressure=blood_pressure,
                        old_peak=old_peak,
                    )
                    st.write(guidance)
                    st.caption("AI guidance is educational only and is not a medical diagnosis.")
                except Exception as error:
                    st.warning(f"AI guidance could not be generated: {error}")

# ---------------------------------------------------------------------------
# Model performance / comparison section
# ---------------------------------------------------------------------------
st.header("Model Performance")
st.write(
    "The table below compares the four available algorithms. Your prediction uses "
    f"the currently selected **{selected_algorithm}** model."
)

with st.expander("Show model comparison table (recomputed on a fresh test split)"):
    if st.button("Run comparison"):
        with st.spinner("Training and comparing models..."):
            data = load_cleaned_data(DATA_PATH)
            x, y = get_features_and_target(data)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
            results, _ = compare_models(x_train, y_train, x_test, y_test)

            comparison_rows = []
            for name, metrics in results.items():
                comparison_rows.append({
                    "Model": name,
                    "Accuracy": round(metrics['accuracy'], 4),
                    "Precision": round(metrics['precision'], 4),
                    "Recall": round(metrics['recall'], 4),
                    "F1-Score": round(metrics['f1_score'], 4),
                })

            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True)
            st.caption(f"Model currently selected for prediction: {selected_algorithm}")
    else:
        st.info("Click 'Run comparison' to train and compare all four algorithms on a fresh split.")

st.divider()
st.caption(f"Selected algorithm: {selected_algorithm}")
