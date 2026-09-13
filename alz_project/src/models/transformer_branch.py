"""
transformer_branch.py  [our contribution]  [reference: ADDFormer (rkushol/ADDFormer) for the
general idea of a ViT-style branch fused with CNN features for AD MRI classification;
this is an independent, standard ViT-encoder implementation, not copied from that repo]

A small Vision-Transformer-style encoder that patchifies the raw MRI input and models
long-range/global relationships that the CNN's local receptive fields may miss.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from .. import config as cfg


class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.proj(x)  # (B, H/p, W/p, embed_dim)
        b = tf.shape(x)[0]
        h_w = x.shape[1] * x.shape[2]
        x = tf.reshape(x, [b, h_w, x.shape[-1]])
        return x


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads,
                                               dropout=dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


def build_transformer_branch(input_tensor, patch_size=None, embed_dim=None,
                              num_heads=None, num_layers=None, mlp_dim=None):
    """Returns a (B, embed_dim) global feature vector (CLS-token-free — uses mean pooling)."""
    patch_size = patch_size or cfg.TRANSFORMER_PATCH_SIZE
    embed_dim = embed_dim or cfg.TRANSFORMER_EMBED_DIM
    num_heads = num_heads or cfg.TRANSFORMER_NUM_HEADS
    num_layers = num_layers or cfg.TRANSFORMER_NUM_LAYERS
    mlp_dim = mlp_dim or cfg.TRANSFORMER_MLP_DIM

    x = PatchEmbedding(patch_size, embed_dim)(input_tensor)
    num_patches = x.shape[1]
    pos_embed = tf.Variable(
        tf.random.normal([1, num_patches, embed_dim], stddev=0.02),
        trainable=True, name="pos_embed"
    )
    x = x + pos_embed

    for i in range(num_layers):
        x = TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, name=f"transformer_block_{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    global_feat = layers.GlobalAveragePooling1D()(x)
    return global_feat
