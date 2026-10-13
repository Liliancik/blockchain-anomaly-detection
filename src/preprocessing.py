"""Data preprocessing pipeline: missing values, encoding, feature selection, normalization.

Transforms raw data into format suitable for autoencoder:
1. Handle missing values (median for numeric, mode for categorical)
2. Encode categorical features (one-hot encoding)
3. Select features (remove labels)
4. Standardize features to zero mean and unit variance
"""
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with appropriate strategy.
    
    Strategy:
    - Numeric columns: Use MEDIAN (robust to outliers)
    - Categorical columns: Use MODE (most frequent value)
    
    Why these strategies?
    - Median: Single outlier doesn't skew it (unlike mean)
    - Mode: Preserves category distribution (respects frequencies)
    
    Args:
        df: DataFrame with potential missing values
    
    Returns:
        DataFrame with all missing values filled
    """
    # Identify column types
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Fill numeric columns with median of that column
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode (most common value) of that column
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convert categorical features to binary columns (one-hot encoding).
    
    Why one-hot encoding?
    - Neural networks only understand numeric inputs
    - One-hot creates interpretable binary features for each category
    - drop_first=True: Prevents multicollinearity (dummy variable trap)
      Example: {Male, Female} → {Female} (Male implied when Female=0)
    
    Example transformation:
    Input rows:
      Gender="M", Grade="A" → After: Gender_F=0, Grade_B=0, Grade_C=0
      Gender="F", Grade="B" → After: Gender_F=1, Grade_B=1, Grade_C=0
    
    Args:
        df: DataFrame with categorical (object) columns
    
    Returns:
        df_encoded: DataFrame with all categorical columns expanded to binary
        cat_cols: List of original categorical column names
    """
    # Find all text/object columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # Expand each categorical to binary columns, drop first to avoid redundancy
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded, cat_cols


def select_features(df: pd.DataFrame, label_like: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Remove label columns, keep only feature columns.
    
    Why remove labels?
    - Labels are what we're trying to predict (outcomes)
    - Features are inputs we learn from
    - Including labels = data leakage = unfair model advantage
    - Model should detect anomalies from behavior, not outcomes
    
    Example:
    Input columns: GPA, Attendance, Study_Hours, Exam_Score, FinalGrade, PassFail
    Labels to remove: FinalGrade, PassFail
    Output features: GPA, Attendance, Study_Hours, Exam_Score
    
    Args:
        df: DataFrame with all columns (features + labels)
        label_like: List of column names to remove (defaults to common label names)
    
    Returns:
        X: DataFrame with only feature columns
        features: List of feature column names
    """
    # Default label columns to exclude
    if label_like is None:
        label_like = ["FinalGrade", "PassFail", "ExamResult"]
    # Keep columns not in the exclusion list
    features = [col for col in df.columns if col not in label_like]
    return df[features], features


def normalize_features(X: pd.DataFrame) -> Tuple[object, pd.DataFrame]:
    """Standardize features to zero mean, unit variance.
    
    Transformation: x_scaled = (x - mean) / std_dev
    
    Why normalize?
    - Neural networks converge much faster on normalized data
    - Prevents features with large ranges from dominating (e.g., age vs. income)
    - Makes learning rate more stable across all features
    
    Example:
    Before: GPA ∈ [0,4], Study_Hours ∈ [0,200]
      → Study_Hours has 50× larger range, dominates learning
    After: GPA ∈ [-1.5, 2.5], Study_Hours ∈ [-1.2, 2.8]
      → Both features have similar scale, equal importance
    
    CRITICAL: fit_transform(X_train) then transform(X_test)
    - Never fit on test set (prevents data leakage)
    - Both use same mean/std from training distribution
    
    Args:
        X: Features to normalize (typically training data)
    
    Returns:
        scaler: Fitted StandardScaler (save to apply same transformation to test data)
        X_scaled: Normalized features (mean≈0, std≈1)
    """
    # Create scaler and fit on this data
    scaler = StandardScaler()
    # fit_transform: learn mean/std from X, then apply transformation
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled
