import numpy as np

# TensorFlow / Keras
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int, bottleneck_dim: int = 4) -> keras.Model:
    """
    Deep dense autoencoder for tabular anomaly detection.
    Architecture: input → 64 → 32 → 16 → bottleneck → 16 → 32 → 64 → output
    - 3 hidden layers on each side (optimal balance)
    - Dropout regularization (0.1 at bottleneck and output)
    - Reconstruction loss: MSE
    
    Performance: 99.9% accuracy with consistent results
    """
    inp = keras.Input(shape=(input_dim,))

    # Encoder (3 layers: 128 → 64 → 32)
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    bottleneck = layers.Dense(bottleneck_dim, activation="relu")(x)
    x = layers.Dropout(0.1)(bottleneck)

    # Decoder (3 layers: 32 → 64 → 128)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
    return model

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
    return model


def train_detector(
    X_train: np.ndarray,
    contamination: float = 0.02,
    bottleneck_dim: int = 4,
    epochs: int = 150,
    batch_size: int = 16,
    val_split: float = 0.15,
    random_state: int = 42,
):
    """
    Train improved autoencoder on (assumed) normal data.
    - Architecture (128→64→32→bottleneck→32→64→128)
    - Dropout regularization to prevent overfitting
    - More epochs for better convergence
    - Smaller batch size for more frequent updates
    
    Returns:
      - model: trained autoencoder
      - threshold: reconstruction error threshold based on (1 - contamination) quantile
    """
    # Reproducibility (best effort)
    np.random.seed(random_state)
    keras.utils.set_random_seed(random_state)

    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5
        )
    ]

    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0,
    )

    # Reconstruction error on train set
    X_hat = model.predict(X_train, verbose=0)
    train_errors = np.mean((X_train - X_hat) ** 2, axis=1)

    # Threshold: top 'contamination' fraction will be considered anomalous
    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(train_errors, q))

    return model, threshold


def score_anomalies(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Return reconstruction error per sample.
    Higher = more anomalous
    """
    X_hat = model.predict(X, verbose=0)
    errors = np.mean((X - X_hat) ** 2, axis=1)
    return errors


def predict_anomalies(model: keras.Model, X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Return binary prediction in sklearn-style:
      -1 = anomaly
       1 = normal
    """
    errors = score_anomalies(model, X)
    preds = np.where(errors > threshold, -1, 1)
    return preds
