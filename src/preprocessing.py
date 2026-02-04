import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded, cat_cols


def select_features(df: pd.DataFrame, label_like: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    if label_like is None:
        label_like = ["FinalGrade", "PassFail", "ExamResult"]
    features = [col for col in df.columns if col not in label_like]
    return df[features], features


def normalize_features(X: pd.DataFrame) -> Tuple[object, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled
