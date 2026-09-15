"""
train_model.py
---------------
Loads the cleaned dataset, performs the same train-test split used in the
notebook, trains and compares Logistic Regression, KNN, SVC, and XGBoost,
prints a comparison of their performance, and saves the selected model
(SVC with kernel='linear', C=1) to models/heart_disease_model.pkl.

Run this once before starting the Streamlit app:
    python train_model.py
"""

import os
import joblib
from sklearn.model_selection import train_test_split

from backend.preprocessing import load_cleaned_data, get_features_and_target
from backend.model import get_selected_model, train_model, evaluate_model, compare_models

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'heart_disease_cleaned.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_model.pkl')


def main():
    print("=" * 60)
    print("Heart Disease Prediction - Model Training")
    print("=" * 60)

    # 1. Load the cleaned dataset (already preprocessed as in the notebook)
    print(f"\n[1/5] Loading cleaned dataset from: {DATA_PATH}")
    data = load_cleaned_data(DATA_PATH)
    print(f"      Dataset shape: {data.shape}")

    # 2. Select features and target (same as notebook)
    print("\n[2/5] Selecting features and target...")
    x, y = get_features_and_target(data)
    print(f"      Features: {list(x.columns)}")
    print(f"      Target: Heart_Disease")

    # 3. Train-test split (same test_size=0.3 as notebook)
    print("\n[3/5] Performing train-test split (test_size=0.3)...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print(f"      Train size: {len(x_train)}, Test size: {len(x_test)}")

    # 4. Train and compare all four algorithms
    print("\n[4/5] Training and comparing models: Logistic Regression, KNN, SVC, XGBoost...")
    results, fitted_models = compare_models(x_train, y_train, x_test, y_test)

    print("\n      Model Comparison (on test set):")
    print("      " + "-" * 70)
    print(f"      {'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("      " + "-" * 70)
    for name, metrics in results.items():
        print(f"      {name:<22}{metrics['accuracy']:<12.4f}{metrics['precision']:<12.4f}"
              f"{metrics['recall']:<12.4f}{metrics['f1_score']:<12.4f}")
    print("      " + "-" * 70)

    for name, metrics in results.items():
        print(f"\n      Classification Report - {name} (test set):")
        print(metrics['report'])

    # 5. Train the selected model (SVC kernel='linear', C=1) and save it
    print("[5/5] Training selected model: SVC(kernel='linear', C=1) and saving...")
    selected_model = get_selected_model()
    selected_model = train_model(selected_model, x_train, y_train)

    train_metrics = evaluate_model(selected_model, x_train, y_train)
    test_metrics = evaluate_model(selected_model, x_test, y_test)

    print("\n      Selected Model (SVC linear) - Train set report:")
    print(train_metrics['report'])
    print("      Selected Model (SVC linear) - Test set report:")
    print(test_metrics['report'])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    print(f"      Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Training complete. You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
