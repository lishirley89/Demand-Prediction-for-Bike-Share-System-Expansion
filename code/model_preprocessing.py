#!/usr/bin/env python3
"""
Preprocessing script: clean and standardize training and test datasets.

- Loads result/training_dataset.csv and result/test_dataset.csv
- Drops rows with any NaN values in numeric feature columns (records how many removed)
- Creates one-hot vectors for the month column
- Standardizes numeric feature columns (fit scaler on training, apply to test)
- Leaves identifier columns (station_id, year) and targets untouched
- Saves preprocessed datasets and a summary report

Outputs:
- result/training_dataset_preprocessed.csv
- result/test_dataset_preprocessed.csv
- result/preprocess_report.txt
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "result/training_dataset.csv"
TEST_PATH = "result/test_dataset.csv"
OUT_DIR = "result"

ID_COLS = ["station_id", "year"]  # Removed month since it will be one-hot encoded
# Targets present in the dataset (kept but not standardized)
TARGETS = [
    "cbike_start",
    "cbike_end",
    "ebike_start",
    "ebike_end",
    "total_start",
    "total_end",
]


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Training or test dataset not found in result/ directory.")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def create_month_onehot(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one-hot encoded columns for month from both training and test datasets.
    Ensures both datasets have the same month columns.
    """
    # Get all unique months from both datasets
    all_months = sorted(set(train_df['month'].unique()) | set(test_df['month'].unique()))
    
    # Create one-hot encoding for training data
    train_month_dummies = pd.get_dummies(train_df['month'], prefix='month', dtype=int)
    # Ensure all month columns exist (fill missing with 0)
    for month in all_months:
        col_name = f'month_{month}'
        if col_name not in train_month_dummies.columns:
            train_month_dummies[col_name] = 0
    
    # Create one-hot encoding for test data
    test_month_dummies = pd.get_dummies(test_df['month'], prefix='month', dtype=int)
    # Ensure all month columns exist (fill missing with 0)
    for month in all_months:
        col_name = f'month_{month}'
        if col_name not in test_month_dummies.columns:
            test_month_dummies[col_name] = 0
    
    # Reorder columns to ensure consistency
    month_cols = sorted([col for col in train_month_dummies.columns if col.startswith('month_')])
    train_month_dummies = train_month_dummies[month_cols]
    test_month_dummies = test_month_dummies[month_cols]
    
    # Drop original month column and concatenate one-hot columns
    train_df_encoded = pd.concat([train_df.drop('month', axis=1), train_month_dummies], axis=1)
    test_df_encoded = pd.concat([test_df.drop('month', axis=1), test_month_dummies], axis=1)
    
    return train_df_encoded, test_df_encoded, month_cols


def select_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = set(ID_COLS + TARGETS)
    # Also exclude month one-hot columns from standardization
    month_cols = [col for col in df.columns if col.startswith('month_')]
    drop_cols.update(month_cols)
    
    numeric_cols = [
        c for c in df.columns
        if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    return numeric_cols


def drop_na_rows(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int]:
    initial = len(df)
    df_clean = df.dropna(subset=cols)
    removed = initial - len(df_clean)
    return df_clean, removed


def standardize_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_df.loc[:, cols] = scaler.fit_transform(train_df[cols])
    test_df.loc[:, cols] = scaler.transform(test_df[cols])
    return train_df, test_df


def main():
    ensure_out_dir()
    train_df, test_df = load_datasets()

    # 1) Create one-hot encoding for month column
    print("Creating one-hot encoding for month column...")
    train_df_encoded, test_df_encoded, month_cols = create_month_onehot(train_df, test_df)
    print(f"Created {len(month_cols)} month one-hot columns: {', '.join(month_cols)}")

    # 2) Determine numeric feature columns to clean/standardize
    feature_cols = select_numeric_feature_columns(train_df_encoded)

    # 3) Drop rows with NaNs (in selected feature columns)
    train_df_clean, train_removed = drop_na_rows(train_df_encoded, feature_cols)
    test_df_clean, test_removed = drop_na_rows(test_df_encoded, feature_cols)

    # 4) Standardize numeric feature columns (fit on training)
    train_std, test_std = standardize_columns(train_df_clean.copy(), test_df_clean.copy(), feature_cols)

    # Save outputs
    train_out = os.path.join(OUT_DIR, "training_dataset_preprocessed.csv")
    test_out = os.path.join(OUT_DIR, "test_dataset_preprocessed.csv")
    train_std.to_csv(train_out, index=False)
    test_std.to_csv(test_out, index=False)

    # Write report
    report_path = os.path.join(OUT_DIR, "preprocess_report.txt")
    with open(report_path, "w") as f:
        f.write("Preprocessing Report\n")
        f.write("====================\n\n")
        f.write(f"Original training rows: {len(train_df)}\n")
        f.write(f"Removed training rows due to NaNs: {train_removed}\n")
        f.write(f"Final training rows: {len(train_df_clean)}\n\n")
        f.write(f"Original test rows: {len(test_df)}\n")
        f.write(f"Removed test rows due to NaNs: {test_removed}\n")
        f.write(f"Final test rows: {len(test_df_clean)}\n\n")
        f.write(f"Month one-hot columns ({len(month_cols)}): {', '.join(month_cols)}\n")
        f.write(f"Standardized columns ({len(feature_cols)}): {', '.join(feature_cols)}\n")

    print(f"Saved preprocessed training data to: {train_out}")
    print(f"Saved preprocessed test data to: {test_out}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
