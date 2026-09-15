"""
model.py
--------
Model training and comparison logic, extracted from the notebook.

The notebook trained/compared:
    - Logistic Regression
    - KNN (n_neighbors=7)
    - SVC (kernel='linear', C=1)   <-- the one actually selected
    - XGBoost (n_estimators=100)

and evaluated each with classification_report on train and test sets.
This file wraps that into reusable functions used by train_model.py.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def get_candidate_models():
    """Return the same set of candidate models compared in the notebook."""
    return {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'SVC': SVC(kernel='linear', C=1),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss'),
    }


def get_selected_model():
    """
    Return the exact model selected in the notebook for the final
    prediction: SVC with kernel='linear' and C=1.
    """
    return SVC(kernel='linear', C=1)


def train_model(model, x_train, y_train):
    """Fit a model on the training data (mirrors model.fit(...) in notebook)."""
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x, y):
    """
    Evaluate a fitted model on a given feature/target set and return a
    dict of metrics plus the full classification report text, matching
    the metrics used in the notebook (classification_report) plus the
    summary metrics requested (accuracy, precision, recall, F1).
    """
    predictions = model.predict(x)
    report_text = classification_report(y, predictions)

    metrics = {
        'accuracy': accuracy_score(y, predictions),
        'precision': precision_score(y, predictions, zero_division=0),
        'recall': recall_score(y, predictions, zero_division=0),
        'f1_score': f1_score(y, predictions, zero_division=0),
        'report': report_text,
    }
    return metrics


def compare_models(x_train, y_train, x_test, y_test):
    """
    Train and evaluate every candidate model on the test set, returning a
    dict of {model_name: metrics}. Used to build the comparison table
    shown in the Streamlit app and printed by train_model.py.
    """
    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        trained = train_model(model, x_train, y_train)
        metrics = evaluate_model(trained, x_test, y_test)
        results[name] = metrics
        fitted_models[name] = trained

    return results, fitted_models
