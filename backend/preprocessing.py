"""
preprocessing.py
-----------------
Reusable data-cleaning functions extracted directly from the original
notebook (ML_Project.ipynb). The logic here is NOT redesigned -- each
function corresponds to a group of cells from the notebook, just wrapped
into functions so they can be reused by train_model.py instead of living
only inside the notebook.

Original notebook flow that this file reproduces:
    1. Rename messy column names
    2. Clean the Age column (replace junk strings, fill NaN, cast to int,
       remove negative signs, replace 0 with median)
    3. Convert Heart_Disease from Yes/No to 1/0
    4. Fill missing values in Cholesterol, Max_Heart_Rate, Blood_Pressure,
       Old_Peak using median
    5. Drop duplicate rows
    6. Clean the Gender column (strip, capitalize, map to 0/1)
    7. Remove outliers using the IQR method for Cholesterol, Max_Heart_Rate,
       Blood_Pressure, Old_Peak
"""

import pandas as pd
import numpy as np


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the messy raw column names to clean, consistent names."""
    data = data.rename(columns={
        'AGE': 'Age',
        'Cholesterol ': 'Cholesterol',
        ' max_heart_rate': 'Max_Heart_Rate',
        'blood_pressure': 'Blood_Pressure',
        'OldPeak': 'Old_Peak'
    })
    return data


def clean_age_column(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the Age column: replace junk values, fill NaN, cast to int,
    remove negative signs, and replace 0 with the median age."""
    data['Age'] = data['Age'].replace('??', 0)
    data['Age'] = data['Age'].replace('--', 0)
    data['Age'] = data['Age'].replace('unknown', 0)
    data['Age'] = data['Age'].fillna(0)
    data['Age'] = data['Age'].astype(int)
    data['Age'] = data['Age'].abs()
    data['Age'] = data['Age'].replace(0, int(data['Age'].median()))
    return data


def clean_heart_disease_column(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the Heart_Disease target column from Yes/No text to 1/0 ints."""
    data['Heart_Disease'] = data['Heart_Disease'].replace('Yes', '1')
    data['Heart_Disease'] = data['Heart_Disease'].replace('No', '0')
    data['Heart_Disease'] = data['Heart_Disease'].astype(int)
    return data


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values using the median, same as the notebook."""
    data['Cholesterol'] = data['Cholesterol'].fillna(data['Cholesterol'].median())
    data['Max_Heart_Rate'] = data['Max_Heart_Rate'].fillna(data['Max_Heart_Rate'].median())
    data['Blood_Pressure'] = data['Blood_Pressure'].fillna(data['Blood_Pressure'].median())
    data['Old_Peak'] = data['Old_Peak'].fillna(data['Old_Peak'].median())
    return data


def drop_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from the dataset."""
    data = data.drop_duplicates()
    return data


def clean_gender_column(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the Gender column: strip whitespace, capitalize, and map to 0/1."""
    data['Gender'] = data['Gender'].str.strip()
    data['Gender'] = data['Gender'].str.capitalize()
    data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
    return data


def remove_cholesterol_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove Cholesterol outliers using the manual IQR method from the
    notebook (median-of-halves quartiles on the sorted column)."""
    c = data[['Cholesterol']]
    c = c.sort_values('Cholesterol')

    half = len(c) // 2
    f_q1 = c.head(half)
    f_q3 = c.tail(half)

    q1 = f_q1['Cholesterol'].median()
    q3 = f_q3['Cholesterol'].median()
    iqr = q3 - q1

    q_max = q3 + (iqr * 1.5)
    q_min = q1 - (iqr * 1.5)

    data = data[(data['Cholesterol'] >= q_min) & (data['Cholesterol'] <= q_max)]
    return data


def remove_max_heart_rate_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove Max_Heart_Rate outliers using the fixed bounds observed from
    the notebook's boxplot inspection (98 - 206)."""
    data = data[(data['Max_Heart_Rate'] >= 98) & (data['Max_Heart_Rate'] <= 206)]
    return data


def remove_blood_pressure_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove Blood_Pressure outliers using the standard IQR method."""
    q1 = data['Blood_Pressure'].quantile(0.25)
    q3 = data['Blood_Pressure'].quantile(0.75)
    iqr = q3 - q1

    q_min = q1 - 1.5 * iqr
    q_max = q3 + 1.5 * iqr

    data = data[(data['Blood_Pressure'] >= q_min) & (data['Blood_Pressure'] <= q_max)]
    return data


def remove_old_peak_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove Old_Peak outliers using the standard IQR method."""
    q1 = data['Old_Peak'].quantile(0.25)
    q3 = data['Old_Peak'].quantile(0.75)
    iqr = q3 - q1

    q_min = q1 - 1.5 * iqr
    q_max = q3 + 1.5 * iqr

    data = data[(data['Old_Peak'] >= q_min) & (data['Old_Peak'] <= q_max)]
    return data


def preprocess_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full original preprocessing pipeline on a raw/messy dataset,
    exactly in the order the notebook performed it. Only use this if you
    are starting from the raw 'heart_disease_messy.csv' file.

    If you already have the cleaned CSV (data/heart_disease_cleaned.csv),
    you do not need this function -- just load it directly with
    load_cleaned_data().
    """
    data = rename_columns(data)
    data = clean_age_column(data)
    data = clean_heart_disease_column(data)
    data = fill_missing_values(data)
    data = drop_duplicate_rows(data)
    data = clean_gender_column(data)
    data = remove_cholesterol_outliers(data)
    data = remove_max_heart_rate_outliers(data)
    data = remove_blood_pressure_outliers(data)
    data = remove_old_peak_outliers(data)
    return data


def load_cleaned_data(csv_path: str) -> pd.DataFrame:
    """Load the already-cleaned dataset (data/heart_disease_cleaned.csv)."""
    return pd.read_csv(csv_path)


# The exact feature columns and their order, as used in the notebook.
FEATURE_COLUMNS = ['Age', 'Cholesterol', 'Max_Heart_Rate', 'Blood_Pressure', 'Old_Peak']
TARGET_COLUMN = 'Heart_Disease'


def get_features_and_target(data: pd.DataFrame):
    """Return (X, y) using the exact feature order from the notebook."""
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return x, y
