"""Variational Autoencoder (VAE) baseline for unsupervised anomaly detection.

The VAE is a generative model that learns the probability distribution
of normal data. It maps inputs to a normal distribution in a latent space,
enforcing it via KL divergence.

Anomalies tend to reconstruct poorly because they do not follow the
distribution learned from normal training data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


class Sampling(layers.Layer):
    """Sample from the latent distribution using the reparameterization trick."""

    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(
            shape=(batch, dim)
        )

        return (
            z_mean
            + tf.exp(0.5 * z_log_var) * epsilon
        )


class VAE(keras.Model):
    """Variational Autoencoder with custom training and validation steps."""

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """Perform one VAE training step."""

        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(
                inputs,
                training=True
            )

            reconstruction = self.decoder(
                z,
                training=True
            )

            # Reconstruction loss.
            # Sum squared reconstruction error across features,
            # then average across the batch.
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(
                        inputs - reconstruction
                    ),
                    axis=-1
                )
            )

            # KL divergence.
            kl_loss = -0.5 * (
                1
                + z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var)
            )

            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    kl_loss,
                    axis=1
                )
            )

            total_loss = (
                reconstruction_loss
                + kl_loss
            )

        grads = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(
                grads,
                self.trainable_weights
            )
        )

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """Compute VAE losses on validation data without updating weights."""

        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data

        # Obtain the latent distribution.
        z_mean, z_log_var, _ = self.encoder(
            inputs,
            training=False
        )

        # Use the latent mean for deterministic validation.
        reconstruction = self.decoder(
            z_mean,
            training=False
        )

        # Reconstruction loss.
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(
                    inputs - reconstruction
                ),
                axis=-1
            )
        )

        # KL divergence.
        kl_loss = -0.5 * (
            1
            + z_log_var
            - tf.square(z_mean)
            - tf.exp(z_log_var)
        )

        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_loss,
                axis=1
            )
        )

        total_loss = (
            reconstruction_loss
            + kl_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs):
        """Return deterministic reconstruction using latent mean."""

        z_mean, _, _ = self.encoder(
            inputs,
            training=False
        )

        return self.decoder(
            z_mean,
            training=False
        )


def build_vae(
    input_dim: int,
    latent_dim: int = 4
) -> VAE:
    """Build and compile the VAE architecture.

    Architecture is aligned with the deterministic Autoencoder
    baseline where possible.

    Args:
        input_dim: Number of input features.
        latent_dim: Number of latent dimensions.

    Returns:
        Compiled VAE model.
    """

    # ==========================================================
    # ENCODER
    # ==========================================================

    encoder_inputs = keras.Input(
        shape=(input_dim,)
    )

    x = layers.Dense(
        128,
        activation="relu"
    )(encoder_inputs)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    # Latent distribution parameters.
    z_mean = layers.Dense(
        latent_dim,
        name="z_mean"
    )(x)

    z_log_var = layers.Dense(
        latent_dim,
        name="z_log_var"
    )(x)

    z = Sampling()(
        [z_mean, z_log_var]
    )

    encoder = keras.Model(
        encoder_inputs,
        [
            z_mean,
            z_log_var,
            z
        ],
        name="encoder"
    )

    # ==========================================================
    # DECODER
    # ==========================================================

    latent_inputs = keras.Input(
        shape=(latent_dim,)
    )

    x = layers.Dense(
        32,
        activation="relu"
    )(latent_inputs)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.1
    )(x)

    decoder_outputs = layers.Dense(
        input_dim,
        activation="linear"
    )(x)

    decoder = keras.Model(
        latent_inputs,
        decoder_outputs,
        name="decoder"
    )

    # ==========================================================
    # VAE
    # ==========================================================

    vae = VAE(
        encoder,
        decoder
    )

    vae.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=5e-4
        )
    )

    return vae


def train_vae(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    contamination: float = config.INJECTION_FRAC,
    latent_dim: int = config.BOTTLENECK_DIM,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = 0.15,
    random_state: int = config.SEED,
) -> tuple:
    """Train the VAE on clean training data.

    When an explicit validation set is supplied, it is used for
    validation loss and early stopping.

    Args:
        X_train:
            Clean training data.

        X_val:
            Clean validation data used for validation loss
            and early stopping.

        contamination:
            Expected anomaly fraction used only for the
            training-derived fallback threshold.

        latent_dim:
            Dimensionality of the latent distribution.

        epochs:
            Maximum number of training epochs.

        batch_size:
            Training batch size.

        val_split:
            Fallback internal validation fraction if X_val
            is not provided.

        random_state:
            Random seed.

    Returns:
        vae:
            Trained VAE model.

    """

    np.random.seed(
        random_state
    )

    tf.random.set_seed(
        random_state
    )

    vae = build_vae(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim
    )

    # ==========================================================
    # EARLY STOPPING
    # ==========================================================

    if X_val is not None:

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                min_delta=1e-5,
                mode="min",
            )
        ]

        validation_kwargs = {
            "validation_data": (
                X_val,
                X_val
            )
        }

    else:

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=1e-4,
                mode="min",
            )
        ]

        validation_kwargs = {
            "validation_split": val_split
        }

    # ==========================================================
    # TRAIN MODEL
    # ==========================================================

    vae.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
        **validation_kwargs,
    )

    # ==========================================================
    # TRAINING-DERIVED FALLBACK THRESHOLD
    # ==========================================================

    X_hat = vae.predict(
        X_train,
        verbose=0
    )

    train_errors = np.mean(
        (X_train - X_hat) ** 2,
        axis=1
    )

    q = max(
        0.0,
        min(
            1.0,
            1.0 - contamination
        )
    )

    threshold = float(
        np.quantile(
            train_errors,
            q
        )
    )

    return vae, threshold


def score_anomalies_vae(
    model: VAE,
    X: np.ndarray
) -> np.ndarray:
    """Compute reconstruction-error anomaly scores.

    Higher scores indicate greater anomaly severity.

    Args:
        model:
            Trained VAE.

        X:
            Data to score.

    Returns:
        One reconstruction-error score per sample.
    """

    X_hat = model.predict(
        X,
        verbose=0
    )

    errors = np.mean(
        (X - X_hat) ** 2,
        axis=1
    )

    return errors


def predict_anomalies_vae(
    model: VAE,
    X: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Predict anomaly labels using a fixed threshold.

    Args:
        model:
            Trained VAE.

        X:
            Data to classify.

        threshold:
            Fixed reconstruction-error threshold.

    Returns:
        Predictions where:
            -1 = anomaly
             1 = normal
    """

    errors = score_anomalies_vae(
        model,
        X
    )

    preds = np.where(
        errors > threshold,
        -1,
        1
    )

    return preds


def optimize_threshold_vae(
    model: VAE,
    X_eval: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1"
) -> tuple:
    """Select a decision threshold using labelled validation data.

    This function should be used on validation data only when
    producing final held-out test results.

    Args:
        model:
            Trained VAE.

        X_eval:
            Validation/evaluation data used for threshold selection.

        y_true:
            Ground-truth labels:
                1 = normal
               -1 = anomaly

        metric:
            Optimization metric:
            'f1', 'precision', 'recall', or 'balanced'.

    Returns:
        Tuple containing:
            best_threshold
            best_score
    """

    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )

    errors = score_anomalies_vae(
        model,
        X_eval
    )

    thresholds = np.sort(
        errors
    )

    best_threshold = thresholds[0]
    best_score = -1

    for threshold in thresholds:

        preds = np.where(
            errors > threshold,
            -1,
            1
        )

        if metric == "f1":

            score = f1_score(
                y_true,
                preds,
                pos_label=-1,
                zero_division=0
            )

        elif metric == "precision":

            score = precision_score(
                y_true,
                preds,
                pos_label=-1,
                zero_division=0
            )

        elif metric == "recall":

            score = recall_score(
                y_true,
                preds,
                pos_label=-1,
                zero_division=0
            )

        elif metric == "balanced":

            prec = precision_score(
                y_true,
                preds,
                pos_label=-1,
                zero_division=0
            )

            rec = recall_score(
                y_true,
                preds,
                pos_label=-1,
                zero_division=0
            )

            score = (
                prec + rec
            ) / 2

        else:

            raise ValueError(
                f"Unknown metric: {metric}"
            )

        if score > best_score:

            best_score = score
            best_threshold = threshold

    return (
        best_threshold,
        best_score
    )