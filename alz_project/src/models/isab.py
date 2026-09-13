"""
isab.py  [Tripathy paper]

Improved Spatial Attention Block (I-SAB).

The paper's I-SAB refines standard spatial attention (CBAM-style channel-pool -> 7x7 conv ->
sigmoid) by using multi-scale pooling (parallel avg/max pooling at multiple kernel sizes) before
the convolution, giving the attention map awareness of context at more than one receptive field
before it's applied. This reimplementation follows that description; exact hyperparameters
(kernel sizes, number of scales) are not fully specified numerically in the paper, so sane
defaults are used below and exposed as constructor args for you to tune/ablate.
"""
import tensorflow as tf
from tensorflow.keras import layers


class ImprovedSpatialAttention(layers.Layer):
    def __init__(self, pool_kernel_sizes=(3, 5, 7), conv_kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.pool_kernel_sizes = pool_kernel_sizes
        self.conv_kernel_size = conv_kernel_size
        self.avg_pools = [layers.AveragePooling2D(pool_size=k, strides=1, padding="same")
                           for k in pool_kernel_sizes]
        self.max_pools = [layers.MaxPooling2D(pool_size=k, strides=1, padding="same")
                           for k in pool_kernel_sizes]
        self.conv = layers.Conv2D(1, conv_kernel_size, padding="same", activation="sigmoid")

    def call(self, x):
        channel_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        channel_max = tf.reduce_max(x, axis=-1, keepdims=True)
        multiscale_feats = [channel_avg, channel_max]
        for avg_pool, max_pool in zip(self.avg_pools, self.max_pools):
            multiscale_feats.append(avg_pool(channel_avg))
            multiscale_feats.append(max_pool(channel_max))
        concat = tf.concat(multiscale_feats, axis=-1)
        attention_map = self.conv(concat)
        return x * attention_map

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pool_kernel_sizes": self.pool_kernel_sizes,
                    "conv_kernel_size": self.conv_kernel_size})
        return cfg
