# Heart Disease Prediction App

A simple end-to-end Machine Learning application that predicts whether a
patient is likely to have heart disease, based on **Age, Cholesterol,
Maximum Heart Rate, Blood Pressure, and Old Peak**. This project is a
direct conversion of an existing Jupyter Notebook (`ML_Project.ipynb`)
into a working **Streamlit + Python** application. The original ML logic,
preprocessing steps, feature selection, and model were kept as close to
the original notebook as possible — this is a refactor into reusable
files, not a redesign.

---

## Features

- Clean Streamlit UI for entering patient information
- Loads an already-trained model — **does not retrain on every prediction**
- Sidebar selector for Logistic Regression, KNN, SVC, and XGBoost
- Simple input validation (rejects negative or unrealistic values)
- Model performance comparison table (Logistic Regression, KNN, SVC, XGBoost)
- Clear, human-readable prediction result

---

## Technologies Used

- Python
- pandas, numpy — data handling
- scikit-learn — Logistic Regression, KNN, SVC, train/test split, metrics
- xgboost — XGBoost classifier
- joblib — saving/loading the trained model
- Streamlit — frontend
- matplotlib, seaborn — visualization (used during data exploration in the notebook)

---

## Project Structure

```
heart-disease-prediction/
├── app.py                        # Streamlit frontend
├── train_model.py                # Trains, compares, and saves the model
├── requirements.txt
├── README.md
├── backend/
│   ├── __init__.py
│   ├── preprocessing.py          # Reusable data cleaning functions
│   ├── model.py                  # Model training / comparison / evaluation
│   └── prediction.py             # Loads saved model + prediction function
├── data/
│   └── heart_disease_cleaned.csv # Cleaned dataset
├── models/
│   └── heart_disease_model.pkl   # Trained model (created by train_model.py)
└── notebooks/
    └── ML_Project.ipynb          # Original notebook, kept as reference
```

---

## Preprocessing Steps (from the original notebook)

All of these were implemented as reusable functions in `backend/preprocessing.py`:

1. Rename messy raw column names to consistent names
2. Clean the `Age` column — replace junk strings (`??`, `--`, `unknown`),
   fill missing values, cast to int, remove negative signs, replace `0`
   with the median age
3. Convert `Heart_Disease` from `Yes`/`No` text to `1`/`0`
4. Fill missing values in `Cholesterol`, `Max_Heart_Rate`,
   `Blood_Pressure`, `Old_Peak` using the median
5. Drop duplicate rows
6. Clean the `Gender` column — strip whitespace, capitalize, map to `0`/`1`
7. Remove outliers using the **IQR method** for `Cholesterol`,
   `Max_Heart_Rate`, `Blood_Pressure`, and `Old_Peak`

> The provided `data/heart_disease_cleaned.csv` is already the output of
> this pipeline, so `train_model.py` loads it directly. If you ever want
> to start from the raw/messy CSV instead, `preprocess_raw_data()` in
> `backend/preprocessing.py` reproduces the full pipeline above.

## Features Used

`Age`, `Cholesterol`, `Max_Heart_Rate`, `Blood_Pressure`, `Old_Peak`
(in this exact order) → target: `Heart_Disease`

## Algorithms Compared

- Logistic Regression
- KNN (`n_neighbors=7`)
- **SVC (`kernel='linear'`, `C=1`) — the selected model used for prediction**
- XGBoost (`n_estimators=100`)

Train/test split: `test_size=0.3`, same as the notebook.

## Model Evaluation

Each model is evaluated using accuracy, precision, recall, F1-score, and
a full `classification_report` on the test set. The comparison table is
printed by `train_model.py` and is also viewable inside the Streamlit app
under "Model Performance".

---

## Installation

```bash
cd heart-disease-prediction
pip install -r requirements.txt
```

## Training the Model

Run this once (or any time you want to retrain):

```bash
python train_model.py
```

This will:
- Load `data/heart_disease_cleaned.csv`
- Select the same features/target as the notebook
- Split into train/test (`test_size=0.3`)
- Train and compare Logistic Regression, KNN, SVC, and XGBoost
- Print the comparison table and classification reports
- Save the selected SVC model to `models/heart_disease_model.pkl`

## Running the Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`)
in your browser.

## Prediction Flow

```
dataset → existing preprocessing/cleaning → feature selection
→ train-test split → train Logistic Regression, KNN, SVC, XGBoost
→ evaluate & compare → select SVC (linear, C=1) → save model
→ Streamlit frontend → user enters patient info → backend builds
a DataFrame with the exact feature names/order → model.predict()
→ human-readable result ("Heart Disease Detected" / "No Heart Disease Detected")
```

---

## How the Notebook Was Converted

| Notebook section | Converted to |
|---|---|
| Column renaming, Age/Gender/Heart_Disease cleaning, missing-value handling, IQR outlier removal | `backend/preprocessing.py` (one function per cleaning step) |
| Feature/target selection, train_test_split | `backend/preprocessing.py::get_features_and_target` + used in `train_model.py` |
| Model definitions (Logistic Regression, KNN, SVC, XGBoost) and `classification_report` evaluation | `backend/model.py` |
| Model fitting (`model.fit(...)`) and saving | `train_model.py`, which saves via `joblib.dump()` |
| Manual prediction call (`model.predict([[60,167,159,178,2.3]])`) | `backend/prediction.py::predict_heart_disease()`, generalized into a reusable function that builds a proper DataFrame with correct column names |
| No frontend existed | `app.py` — new Streamlit UI for input + result display |

### Changes Made to the Original Code (and why)

1. **Wrapped notebook cells into functions** (`backend/preprocessing.py`,
   `backend/model.py`) — necessary so the logic can be imported and reused
   by both `train_model.py` and `app.py`, instead of only existing as
   notebook cells. The underlying logic (order of operations, thresholds,
   formulas) is unchanged.
2. **Prediction input as a DataFrame with named columns** instead of a
   raw list (`model.predict([[60,167,...]])`) — this avoids ambiguity
   about feature order and avoids scikit-learn warnings about missing
   feature names. The feature order (`Age, Cholesterol, Max_Heart_Rate,
   Blood_Pressure, Old_Peak`) is identical to the notebook.
2. **Model is trained once and saved with `joblib`**, then loaded by the
   app — the notebook trained the model inline; saving/loading was added
   only so the Streamlit app doesn't retrain on every prediction, as required.
3. **Added basic input validation** in `app.py` (rejects negative values
   and unrealistic ages) — this was not present in the notebook (which
   used fixed sample data) but is necessary for a usable app with
   user-entered input.
4. **No new preprocessing techniques were introduced** — no
   `StandardScaler`, no `Pipeline`, no scaling. The model is trained on
   raw feature values exactly as in the notebook.

The original notebook (`notebooks/ML_Project.ipynb`) is preserved
unchanged and included for reference.

---

## Testing Performed

- ✅ Dataset loads correctly (`data/heart_disease_cleaned.csv`, 2123 rows × 7 columns)
- ✅ `train_model.py` runs successfully: trains and compares all four models,
  prints classification reports, and saves `models/heart_disease_model.pkl`
- ✅ Model comparison table produced correctly
- ✅ Trained model loads correctly via `joblib`
- ✅ `predict_heart_disease()` tested with sample input
  (Age=60, Cholesterol=167, Max_Heart_Rate=159, Blood_Pressure=178, Old_Peak=2.3)
  — returned a valid human-readable result. This sample was used only for
  testing and is **not** hard-coded anywhere in the application.
- ✅ Streamlit app (`streamlit run app.py`) starts successfully and serves
  the page (HTTP 200)

---

## Possible Future Improvements

- Add more features from the original dataset (e.g., Gender) if desired
- Add cross-validation for more robust model comparison
- Add a confusion-matrix / ROC-curve visualization in the app
- Add unit tests for the backend functions
- Containerize with Docker (intentionally left out for this academic/interview-level scope)
