"""
lime_explain.py  [our contribution]  [uses the `lime` library, Ribeiro et al. 2016]

Local, model-agnostic explanations via superpixel perturbation — a useful complement to
Grad-CAM (which is gradient/architecture-dependent) and SHAP (which is additive-feature-based).
"""
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries


def explain_instance(model, image_2d_or_3d: np.ndarray, class_names, num_samples=1000):
    """
    image_2d_or_3d: (H, W) or (H, W, 1) grayscale image, values in [0, 1]
    model: keras model expecting (B, H, W, 1) input
    Returns: lime ImageExplanation object
    """
    if image_2d_or_3d.ndim == 2:
        image_2d_or_3d = image_2d_or_3d[..., np.newaxis]

    # LIME's default image pipeline assumes RGB; we replicate the single grayscale
    # channel to 3 channels for the perturbation/segmentation step, then slice back
    # to 1 channel when calling the real model in predict_fn.
    image_rgb = np.repeat(image_2d_or_3d, 3, axis=-1)

    def predict_fn(images_rgb_batch):
        gray_batch = images_rgb_batch[..., :1]  # take one channel back
        return model.predict(gray_batch, verbose=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb, predict_fn, top_labels=len(class_names),
        hide_color=0, num_samples=num_samples,
    )
    return explanation


def get_explanation_overlay(explanation, label_index, positive_only=True, num_features=8):
    temp, mask = explanation.get_image_and_mask(
        label_index, positive_only=positive_only, num_features=num_features, hide_rest=False
    )
    return mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask)
