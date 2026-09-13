"""utils.py — seeding, plotting, small helpers shared across notebooks/scripts."""
import os
import random
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import config as cfg


def set_seed(seed: int = None):
    seed = seed if seed is not None else cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_environment():
    print("Python / TF check")
    print("-" * 40)
    print(f"TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available      : YES ({len(gpus)} device(s))")
        for g in gpus:
            print(f"  - {g}")
    else:
        print("GPU available      : NO — training will run on CPU and be slow. "
              "Consider Google Colab (free T4 GPU) for the real training runs.")
    return len(gpus) > 0


def plot_training_history(history, title_prefix="", out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history.get("val_loss", []), label="val")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history.get("val_accuracy", []), label="val")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", out_path=None):
    class_names = class_names or cfg.CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return cm


def save_metrics(metrics_dict: dict, experiment_name: str):
    path = os.path.join(cfg.RESULTS_DIR, f"{experiment_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics -> {path}")


def load_metrics(experiment_name: str) -> dict:
    path = os.path.join(cfg.RESULTS_DIR, f"{experiment_name}_metrics.json")
    with open(path) as f:
        return json.load(f)
