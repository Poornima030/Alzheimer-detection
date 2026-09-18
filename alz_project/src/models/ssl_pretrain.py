"""
ssl_pretrain.py  [our contribution]  [reference: carloalbertobarbano/contrastive-learning-brain-mri
for the general SimCLR-on-brain-MRI idea; loss/training loop reimplemented independently here
following Chen et al. 2020 "SimCLR" NT-Xent formulation]

Self-supervised pretraining for the CNN backbone using unlabeled MRI images, before supervised
fine-tuning (Experiment 4). Implements SimCLR-style contrastive learning:

  image -> two augmented views -> shared backbone -> projection head -> NT-Xent loss

After pretraining, `transplant_backbone_weights()` copies the learned backbone weights into a
freshly-built supervised fusion model (experiment="cbam_transformer_ssl") before fine-tuning.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .depthwise_cnn import build_backbone
from .. import config as cfg
from ..augmentation import ssl_augment_pair


def build_ssl_model():
    """Backbone + projection head for contrastive pretraining."""
    inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS), name="ssl_input")
    feature_map, multiscale_pooled = build_backbone(inputs, return_multiscale_features=True)
    projection = tf.keras.Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(cfg.SSL_PROJECTION_DIM),
    ], name="projection_head")(multiscale_pooled)
    projection = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                                name="l2_normalize")(projection)
    backbone_model = Model(inputs, multiscale_pooled, name="ssl_backbone")
    ssl_model = Model(inputs, projection, name="ssl_pretrain_model")
    return ssl_model, backbone_model


def nt_xent_loss(z1, z2, temperature=None):
    """Normalized temperature-scaled cross entropy loss (SimCLR)."""
    temperature = temperature or cfg.SSL_TEMPERATURE
    batch_size = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)  # (2B, D)
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature  # (2B, 2B)

    mask = tf.eye(2 * batch_size, dtype=tf.bool)
    sim_matrix = tf.where(mask, tf.fill(tf.shape(sim_matrix), -1e9), sim_matrix)

    positives = tf.concat([
        tf.range(batch_size, 2 * batch_size),
        tf.range(0, batch_size),
    ], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=positives, logits=sim_matrix)
    return tf.reduce_mean(loss)


def pretrain_ssl(unlabeled_dataset=None, epochs=None, learning_rate=1e-3, verbose=True):
    """
    unlabeled_dataset: kept as a parameter for backward compatibility with existing callers
                        (e.g. train.py), but this implementation rebuilds its own augmented-pair
                        pipeline internally via tf.data (faster than per-batch tf.map_fn calls,
                        and prints step-level progress instead of only per-epoch).
    Returns the pretrained backbone_model (input -> multiscale-pooled features).
    """
    from ..data_loader import get_filepaths_and_labels
    from ..preprocessing import load_and_preprocess_image
    from ..augmentation import ssl_augment_pair

    epochs = epochs or cfg.SSL_EPOCHS
    train_paths, _ = get_filepaths_and_labels("train")

    paths_ds = tf.data.Dataset.from_tensor_slices(train_paths)
    paths_ds = paths_ds.shuffle(buffer_size=len(train_paths), seed=cfg.SEED)
    paths_ds = paths_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    paired_ds = paths_ds.map(ssl_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    paired_ds = paired_ds.batch(cfg.SSL_BATCH_SIZE, drop_remainder=True)
    paired_ds = paired_ds.prefetch(tf.data.AUTOTUNE)

    ssl_model, backbone_model = build_ssl_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def train_step(view1, view2):
        with tf.GradientTape() as tape:
            z1 = ssl_model(view1, training=True)
            z2 = ssl_model(view2, training=True)
            loss = nt_xent_loss(z1, z2)
        grads = tape.gradient(loss, ssl_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, ssl_model.trainable_variables))
        return loss

    steps_per_epoch = len(train_paths) // cfg.SSL_BATCH_SIZE
    history = []
    for epoch in range(epochs):
        losses = []
        for step, (v1, v2) in enumerate(paired_ds):
            loss = train_step(v1, v2)
            losses.append(float(loss))
            if verbose and step % 50 == 0:
                print(f"[SSL] epoch {epoch + 1}/{epochs} step {step}/{steps_per_epoch} "
                      f"loss={float(loss):.4f}")
        mean_loss = sum(losses) / max(len(losses), 1)
        history.append(mean_loss)
        if verbose:
            print(f"[SSL] epoch {epoch + 1}/{epochs} DONE — mean NT-Xent loss = {mean_loss:.4f}")

    return backbone_model, history


def transplant_backbone_weights(pretrained_backbone: Model, target_full_model: Model):
    """
    Copies weights layer-by-layer (matched by name) from a pretrained SSL backbone into a
    freshly built supervised model (e.g. fusion_model.build_model('cbam_transformer_ssl')),
    so fine-tuning starts from SSL-learned representations instead of random init.
    Layers with no name match (e.g. CBAM, transformer branch, final MLP) keep random init,
    which is expected — only the shared CNN backbone was pretrained.
    """
    pretrained_layers = {l.name: l for l in pretrained_backbone.layers}
    n_copied = 0
    for layer in target_full_model.layers:
        if layer.name in pretrained_layers:
            src_weights = pretrained_layers[layer.name].get_weights()
            if src_weights and layer.get_weights() and \
               all(a.shape == b.shape for a, b in zip(src_weights, layer.get_weights())):
                layer.set_weights(src_weights)
                n_copied += 1
    print(f"Transplanted weights for {n_copied} layers from SSL-pretrained backbone.")
    return target_full_model
