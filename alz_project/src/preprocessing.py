"""
preprocessing.py  [Tripathy paper: grayscale MRI, resize, normalize]

Note on skull stripping: the Kaggle OASIS multi-class dataset used here is derived from
pre-processed OASIS-1 slices that are already skull-stripped/registered upstream. We keep
a `skull_strip_placeholder` hook for completeness (matching the architecture diagram) but
it is a no-op by default. If you work with raw OASIS NIfTI volumes instead of this Kaggle
slice dataset, replace it with a real tool (e.g. FSL BET, or a deep-learning skull-stripper
such as HD-BET) before calling this pipeline.
"""
import tensorflow as tf
from . import config as cfg


def skull_strip_placeholder(img: tf.Tensor) -> tf.Tensor:
    """No-op hook — see module docstring. Kept so the pipeline mirrors the architecture diagram."""
    return img


def load_and_preprocess_image(path: tf.Tensor) -> tf.Tensor:
    """
    path: scalar string tensor (file path)
    returns: float32 tensor of shape (IMG_SIZE, IMG_SIZE, CHANNELS), values in [0, 1]
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=cfg.CHANNELS, expand_animations=False)
    img.set_shape([None, None, cfg.CHANNELS])
    img = tf.image.resize(img, [cfg.IMG_SIZE, cfg.IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = skull_strip_placeholder(img)
    # per-image standardization gives a small but consistent boost for MRI intensity
    # variance across scanners/sessions
    img = tf.image.per_image_standardization(img)
    return img
