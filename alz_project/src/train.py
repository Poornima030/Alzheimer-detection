"""
train.py — CLI entry point to train any experiment variant end to end.

Usage:
    python -m src.train --experiment baseline
    python -m src.train --experiment cbam
    python -m src.train --experiment cbam_transformer
    python -m src.train --experiment cbam_transformer_ssl   # runs SSL pretraining first
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

from . import config as cfg
from .utils import set_seed, check_environment, plot_training_history, save_metrics
from .data_loader import get_train_val_test_datasets, get_unlabeled_dataset_for_ssl
from .models.fusion_model import build_model, get_backbone_submodel
from .models import ssl_pretrain as ssl


def train_one_experiment(experiment: str, epochs: int = None, verbose: bool = True):
    set_seed()
    epochs = epochs or cfg.EPOCHS

    train_ds, val_ds, test_ds, class_weights = get_train_val_test_datasets()

    build_experiment = "cbam_transformer" if experiment == "cbam_transformer_ssl" else experiment
    model = build_model(build_experiment)

    if experiment == "cbam_transformer_ssl":
        print("Running SSL pretraining before supervised fine-tuning ...")
        unlabeled_ds = get_unlabeled_dataset_for_ssl()
        pretrained_backbone, ssl_history = ssl.pretrain_ssl(unlabeled_ds)
        ssl.transplant_backbone_weights(pretrained_backbone, model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, f"{experiment}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weights, callbacks=callbacks, verbose=1 if verbose else 2,
    )

    if verbose:
        plot_training_history(history, title_prefix=experiment,
                               out_path=os.path.join(cfg.RESULTS_DIR, f"{experiment}_history.png"))

    metrics = evaluate_on_test(model, test_ds, experiment)
    return model, history, metrics


def evaluate_on_test(model, test_ds, experiment_name: str) -> dict:
    y_true, y_pred, y_prob = [], [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_prob.append(probs)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(y_batch.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # can happen if a class is missing from a tiny test split

    metrics = {
        "experiment": experiment_name,
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_macro": float(auc),
    }
    print(metrics)
    save_metrics(metrics, experiment_name)

    np.save(os.path.join(cfg.RESULTS_DIR, f"{experiment_name}_y_true.npy"), y_true)
    np.save(os.path.join(cfg.RESULTS_DIR, f"{experiment_name}_y_pred.npy"), y_pred)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=cfg.EXPERIMENTS)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    check_environment()
    train_one_experiment(args.experiment, epochs=args.epochs)
