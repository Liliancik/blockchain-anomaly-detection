"""PCA baseline for anomaly detection."""

import numpy as np
from sklearn.decomposition import PCA


def train_pca(X_train, n_components=4):
    """Train PCA on the normal training data."""

    pca = PCA(
        n_components=n_components
    )

    pca.fit(X_train)

    return pca


def score_anomalies_pca(pca, X):
    """Calculate reconstruction error for each sample."""

    # Reduce the data to the PCA components
    X_reduced = pca.transform(X)

    # Reconstruct the original features
    X_reconstructed = pca.inverse_transform(
        X_reduced
    )

    # Mean squared reconstruction error for each sample
    scores = np.mean(
        (X - X_reconstructed) ** 2,
        axis=1
    )

    return scores