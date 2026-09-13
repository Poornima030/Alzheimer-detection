"""
shap_explain.py  [our contribution]  [uses the `shap` library, Lundberg & Lee 2017]

Wraps the trained Keras model with SHAP's GradientExplainer (works well for CNNs,
much faster than KernelExplainer for image inputs).
"""
import numpy as np
import shap


def build_explainer(model, background_images: np.ndarray):
    """
    background_images: (N, H, W, C) small reference set (e.g. 50-100 training images)
                        used as the SHAP baseline distribution.
    """
    return shap.GradientExplainer(model, background_images)


def explain_instance(explainer, image_batch: np.ndarray, num_classes: int):
    """
    image_batch: (1, H, W, C)
    Returns: shap_values, a list of length num_classes, each (1, H, W, C)
    """
    shap_values, _ = explainer.shap_values(image_batch, ranked_outputs=num_classes)
    return shap_values


def plot_shap_summary(shap_values, image_batch, class_names, out_path=None):
    """Saves (or shows) a SHAP image-summary plot for one instance across all classes."""
    shap.image_plot(shap_values, image_batch, show=(out_path is None))
    if out_path:
        import matplotlib.pyplot as plt
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
