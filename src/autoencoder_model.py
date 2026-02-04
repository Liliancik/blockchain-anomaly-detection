from sklearn.ensemble import IsolationForest
import numpy as np


def train_detector(X, random_state: int = 42):
    """Train a lightweight anomaly detector (Isolation Forest)"""
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=random_state)
    clf.fit(X)
    return clf


def score_anomalies(clf, X) -> np.ndarray:
    """Return anomaly scores (the lower, the more anomalous)."""
    # In sklearn the decision_function gives anomaly score (higher = more normal)
    scores = clf.decision_function(X)
    return scores

def predict_anomalies(clf, X) -> np.ndarray:
    """Return binary prediction: -1 anomaly, 1 normal"""
    preds = clf.predict(X)
    return preds
