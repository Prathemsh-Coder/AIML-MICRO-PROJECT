"""
Data loading and preprocessing module for SmartSegment.
Supports preloaded Mall Customers, Online Retail RFM, and user-uploaded datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_preloaded_dataset(dataset_name: str) -> pd.DataFrame:
    """Loads a pre-bundled dataset by name."""
    if dataset_name == "Mall Customers":
        return pd.read_csv(DATA_DIR / "mall_customers.csv")
    elif dataset_name == "Online Retail (RFM)":
        return pd.read_csv(DATA_DIR / "online_retail_rfm.csv")
    else:
        raise ValueError(f"Unknown preloaded dataset: {dataset_name}")

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates structural summary of the dataframe."""
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Potential ID columns (high cardinality or named with ID)
    id_candidates = [col for col in df.columns if 'id' in col.lower() or df[col].nunique() == len(df)]
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "id_candidates": id_candidates,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }

def clean_and_prepare_features(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    id_col: str = None,
    handle_missing: str = "median"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans selected features, imputes missing values, and drops duplicates.
    Returns:
        (clean_df_with_meta, X_features)
    """
    clean_df = df.copy()
    
    # Drop exact duplicates
    clean_df = clean_df.drop_duplicates()
    
    # Impute missing values for feature columns
    for col in feature_cols:
        if clean_df[col].isnull().sum() > 0:
            if handle_missing == "median":
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            elif handle_missing == "mean":
                clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
            else:
                clean_df = clean_df.dropna(subset=[col])
                
    X = clean_df[feature_cols].copy()
    return clean_df, X
