"""
depthwise_cnn.py  [Tripathy paper]

Reimplementation of the base paper's backbone:
  - 10 depthwise-separable convolution layers
  - 4 max-pooling layers interspersed
  - I-SAB attached after every even-numbered conv layer (2, 4, 6, 8, 10)
  - Multiscale feature fusion: the I-SAB-refined outputs from multiple depths are
    global-average-pooled and concatenated before the classifier head
  - 256-unit ReLU MLP -> 4-class softmax

The paper does not publish exact per-layer filter counts/strides, so this uses a standard
progressively-widening filter schedule (32 -> 256) with pooling after every 2-3 conv layers,
which is a common and reasonable choice for a 10-layer depthwise-separable design at 128x128
input. Treat filter counts/pool placement as tunable hyperparameters for your own experiments.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from .isab import ImprovedSpatialAttention
from .. import config as cfg

# (filters, add_pool_after) for each of the 10 depthwise-separable conv layers
_LAYER_SCHEDULE = [
    (32, False), (32, True),     # layer 1-2, pool after 2   -> I-SAB after layer 2
    (64, False), (64, True),     # layer 3-4, pool after 4   -> I-SAB after layer 4
    (128, False), (128, True),   # layer 5-6, pool after 6   -> I-SAB after layer 6
    (128, False), (128, True),   # layer 7-8, pool after 8   -> I-SAB after layer 8
    (256, False), (256, False),  # layer 9-10                -> I-SAB after layer 10
]


def build_backbone(input_tensor, return_multiscale_features=True):
    """
    Returns:
      final_feature_map: the last conv block's output (for CBAM/Transformer branches to consume)
      multiscale_pooled: concatenated GAP features from each I-SAB stage (Tripathy fusion), or None
    """
    x = input_tensor
    multiscale_feats = []

    for i, (filters, pool_after) in enumerate(_LAYER_SCHEDULE, start=1):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        if i % 2 == 0:  # even-numbered layers get I-SAB, per the paper
            x = ImprovedSpatialAttention(name=f"isab_layer{i}")(x)
            if return_multiscale_features:
                multiscale_feats.append(layers.GlobalAveragePooling2D()(x))

        if pool_after:
            x = layers.MaxPooling2D(pool_size=2)(x)

    multiscale_pooled = layers.Concatenate(name="multiscale_fusion")(multiscale_feats) \
        if return_multiscale_features else None

    return x, multiscale_pooled


def build_baseline_model(num_classes=None) -> Model:
    """Experiment 1: pure Tripathy-style baseline (backbone + multiscale fusion + MLP + softmax)."""
    num_classes = num_classes or cfg.NUM_CLASSES
    inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS), name="mri_input")
    _, multiscale_pooled = build_backbone(inputs, return_multiscale_features=True)

    x = layers.Dense(256, activation="relu", name="mlp_256")(multiscale_pooled)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="tripathy_baseline")
