"""
gradcam.py  [our contribution]  [standard Grad-CAM formulation: Selvaraju et al. 2017]

Class-discriminative heatmaps showing which MRI regions drove a prediction.
"""
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if shape is not None and len(shape) == 4:  # (B, H, W, C) — a conv-like feature map
            return layer.name
    raise ValueError("No 4D (conv-like) layer found in model for Grad-CAM.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: (1, H, W, C) preprocessed image batch
    Returns: heatmap as (h, w) numpy array in [0, 1]
    """
    last_conv_layer_name = last_conv_layer_name or find_last_conv_layer_name(model)
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap_on_image(img_2d: np.ndarray, heatmap: np.ndarray, alpha=0.4):
    """
    img_2d: (H, W) grayscale image in [0, 1]
    heatmap: (h, w) Grad-CAM output in [0, 1]
    Returns: (H, W, 3) RGB overlay image, uint8
    """
    h, w = img_2d.shape
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], [h, w]).numpy().squeeze()
    jet = cm.get_cmap("jet")(heatmap_resized)[:, :, :3]
    base_rgb = np.stack([img_2d] * 3, axis=-1)
    overlay = (1 - alpha) * base_rgb + alpha * jet
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay
