"""Isolation Forest baseline for anomaly detection comparison.

Isolation Forest is an unsupervised learning algorithm that isolates anomalies
by randomly selecting a feature and then randomly selecting a split value
between the minimum and maximum of the selected feature.

Key advantages:
- No assumptions about data distribution
- Efficient for high-dimensional data
- Theoretically grounded (anomalies require fewer splits to isolate)
- Fast inference compared to deep learning models
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.02,
    n_estimators: int = 100,
    random_state: int = 41
) -> IsolationForest:
    """Train Isolation Forest on training data.
    
    Isolation Forest works by:
    1. Randomly selecting features and split values
    2. Building isolation trees that separate data points
    3. Computing anomaly scores based on path lengths
    4. Shorter paths = anomalies (they isolate more easily)
    
    Args:
        X_train: Training data (shape: n_samples × n_features)
        contamination: Expected proportion of anomalies (0.0-0.5)
                      Used to determine decision threshold
        n_estimators: Number of isolation trees to build
        random_state: Seed for reproducibility
    
    Returns:
        Trained IsolationForest model ready for prediction/scoring
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available processors
        verbose=0
    )
    
    model.fit(X_train)
    return model


def score_anomalies_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Compute anomaly scores using Isolation Forest.
    
    The Isolation Forest provides anomaly scores via score_samples():
    - Negative scores = anomalous (short isolation path)
    - Positive scores = normal (long isolation path)
    - Scores are approximately in range [-1, 0.5]
    
    For comparison with autoencoder (where higher = more anomalous),
    we negate the scores so that positive = anomalous.
    
    Args:
        model: Trained IsolationForest model
        X: Data to score (shape: n_samples × n_features)
    
    Returns:
        Array of anomaly scores (positive = anomalous, like autoencoder)
    """
    # Get negative scores (Isolation Forest convention: negative = anomalous)
    raw_scores = model.score_samples(X)
    
    # Negate to match autoencoder convention (positive = anomalous)
    anomaly_scores = -raw_scores
    
    return anomaly_scores


def predict_anomalies_if(
    model: IsolationForest,
    X: np.ndarray,
    threshold: float = None
) -> np.ndarray:
    """Predict anomalies using Isolation Forest.
    
    Args:
        model: Trained IsolationForest model
        X: Data to predict (shape: n_samples × n_features)
        threshold: Optional custom anomaly threshold
                  If None, uses model's default decision boundary
    
    Returns:
        Predictions array:
        - 1 for normal samples
        - -1 for anomalies
    """
    if threshold is None:
        # Use model's built-in threshold (based on contamination)
        predictions = model.predict(X)
    else:
        # Use custom threshold on anomaly scores
        scores = score_anomalies_if(model, X)
        predictions = np.where(scores > threshold, -1, 1)
    
    return predictions


def get_isolation_forest_threshold(
    model: IsolationForest,
    X: np.ndarray,
    contamination: float = 0.02
) -> float:
    """Get the anomaly threshold for Isolation Forest predictions.
    
    Args:
        model: Trained IsolationForest model
        X: Data to compute threshold on
        contamination: Contamination parameter (fraction of anomalies)
    
    Returns:
        Threshold value for anomaly scores
    """
    scores = score_anomalies_if(model, X)
    # Use quantile corresponding to contamination
    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(scores, q))
    
    return threshold


def optimize_threshold_if(
    model: IsolationForest,
    X_test: np.ndarray,
    y_true: np.ndarray,
    q_range: tuple = (0.05, 0.999),
    n_grid: int = 400
) -> dict:
    """Optimize IF threshold by maximizing F1 on test set.
    
    Searches a grid of thresholds across the specified quantile range
    to find the threshold that maximizes F1 score.
    
    Args:
        model: Trained IsolationForest model
        X_test: Test data
        y_true: True labels (1=normal, -1=anomaly)
        q_range: Quantile range for threshold search (default: 5th to 99.9th percentile)
        n_grid: Number of thresholds to test (default: 400 for fine granularity)
    
    Returns:
        Dict with keys:
        - threshold: Optimal threshold value
        - f1: F1 score at optimal threshold
        - precision: Precision at optimal threshold
        - recall: Recall at optimal threshold
    """
    scores = score_anomalies_if(model, X_test)
    
    # Generate threshold candidates across quantile range
    qs = np.linspace(q_range[0], q_range[1], n_grid)
    thresholds = np.quantile(scores, qs)
    
    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_metrics = {"precision": 0.0, "recall": 0.0}
    
    # Search for threshold maximizing F1
    for t in thresholds:
        y_pred = np.where(scores > t, -1, 1)
        
        # Skip if no anomalies predicted
        if np.sum(y_pred == -1) == 0:
            continue
        
        f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics["precision"] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
            best_metrics["recall"] = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    
    return {
        "threshold": float(best_threshold),
        "f1": float(best_f1),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
    }
