"""
cbam.py  [our contribution]  [reference: thanhkaist/AttentionResnet, standard CBAM formulation
from Woo et al. 2018 "CBAM: Convolutional Block Attention Module" — reimplemented independently]

Convolutional Block Attention Module: sequential channel attention -> spatial attention.
This is additive to the I-SAB already present in the Tripathy backbone (isab.py); CBAM here
is applied at the feature-fusion stage rather than replacing I-SAB, so Experiment 2 tests
whether channel-attention + I-SAB's spatial attention together beat I-SAB alone.
"""
import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        hidden = max(channels // self.reduction_ratio, 4)
        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(channels),
        ])

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x, axis=[1, 2])
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        scale = tf.nn.sigmoid(avg_out + max_out)
        scale = scale[:, tf.newaxis, tf.newaxis, :]
        return x * scale


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        scale = self.conv(concat)
        return x * scale


class CBAM(layers.Layer):
    def __init__(self, reduction_ratio=8, spatial_kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def call(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
