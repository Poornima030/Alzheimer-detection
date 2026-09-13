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

    Corrupt/unreadable files are replaced with a black image rather than crashing the whole
    training run — with 50k+ real files sourced from Kaggle, a handful of truncated/corrupt
    images is common. A one-off warning is printed via tf.print so you can find and remove
    genuinely bad files afterward if you want a fully clean dataset.
    """
    raw = tf.io.read_file(path)

    def _decode():
        img = tf.io.decode_image(raw, channels=cfg.CHANNELS, expand_animations=False)
        img.set_shape([None, None, cfg.CHANNELS])
        return img

    def _fallback():
        tf.print("[warn] could not decode image, using blank fallback:", path)
        return tf.zeros([cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS], dtype=tf.uint8)

    img = tf.py_function(
        lambda p: _safe_decode(p.numpy()), [path], tf.uint8
    )
    img.set_shape([None, None, cfg.CHANNELS])

    img = tf.image.resize(img, [cfg.IMG_SIZE, cfg.IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = skull_strip_placeholder(img)
    img = tf.image.per_image_standardization(img)
    return img


def _safe_decode(path_bytes):
    """Runs in eager/py_function context so we can use plain Python try/except on real files."""
    import numpy as np
    from PIL import Image
    try:
        img = Image.open(path_bytes.decode("utf-8"))
        img = img.convert("L") if cfg.CHANNELS == 1 else img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        if cfg.CHANNELS == 1 and arr.ndim == 2:
            arr = arr[..., np.newaxis]
        return arr
    except Exception as e:
        print(f"[warn] could not decode '{path_bytes.decode('utf-8', errors='ignore')}': {e} "
              f"— using blank fallback image.")
        return np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS), dtype=np.uint8)
