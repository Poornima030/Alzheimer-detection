"""
data_loader.py  [Tripathy paper: 4-class OASIS setup]  [reference: hack4health_alzheimers
for the general idea of an ID/folder based OASIS loader, reimplemented independently here]

Turns the patient-metadata CSVs + a directory of MRI images into tf.data.Dataset pipelines.

Three supported layouts, auto-detected in this order (see get_filepaths_and_labels):

1. Pre-split class folders (matches the Kaggle "oasis-alzheimers-detection-multi-class-dataset"
   download as-is):
       data/images/train/<ClassName>/*.jpg
       data/images/test/<ClassName>/*.jpg
   Label comes from the folder name. Preferred when present, since it preserves the dataset
   author's original train/test split instead of re-shuffling it.

2. MATCH_BY_ID = True in config.py:
   data/images/<ID><IMG_EXT>   e.g. data/images/OAS1_0001_MR1.jpg
   Label comes from the CSV's `class` column, joined on `ID`.

3. Flat class folders with no existing split (config.MATCH_BY_ID = False and no train/test
   subfolders found):
   data/images/<ClassName>/<anything>.jpg
   Label comes from the folder name; a deterministic 80/20 split is created here.
   Use build_flat_image_index() to convert a raw Kaggle download into this shape first if
   it isn't already train/test split.
"""
import os
import shutil
import glob
import pandas as pd
import numpy as np
import tensorflow as tf

from . import config as cfg
from .preprocessing import load_and_preprocess_image
from .augmentation import augment_image


def build_flat_image_index(raw_dir: str, out_dir: str):
    """
    Helper for datasets that arrive as class-labeled folders instead of <ID>.jpg files.
    Copies/links raw_dir/<ClassName>/*.jpg -> out_dir/<ClassName>/*.jpg unchanged, so
    the class-folder loader (MATCH_BY_ID=False) can consume it directly.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for class_name in cfg.CLASS_NAMES:
        src_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(src_dir):
            print(f"[warn] no folder found for class '{class_name}' under {raw_dir}")
            continue
        dst_dir = os.path.join(out_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for f in glob.glob(os.path.join(src_dir, "*")):
            dst = os.path.join(dst_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
                n += 1
    print(f"Indexed {n} images into {out_dir}")


def _load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["ID", "class"]].dropna(subset=["ID", "class"]).copy()
    df["class"] = df["class"].astype(str).str.strip()
    unknown = sorted(set(df["class"]) - set(cfg.CLASS_NAMES))
    if unknown:
        print(f"[warn] classes in CSV not in config.CLASS_NAMES: {unknown} "
              f"(check spelling/casing against {cfg.CLASS_NAMES})")
    return df


def _resolve_paths_by_id(df: pd.DataFrame):
    """Returns (filepaths, labels, missing_count) for the MATCH_BY_ID=True layout."""
    paths, labels, missing = [], [], 0
    for _, row in df.iterrows():
        candidate = os.path.join(cfg.IMAGES_DIR, row["ID"] + cfg.IMG_EXT)
        if os.path.exists(candidate):
            if row["class"] not in cfg.CLASS_TO_IDX:
                continue
            paths.append(candidate)
            labels.append(cfg.CLASS_TO_IDX[row["class"]])
        else:
            missing += 1
    return paths, labels, missing


def _resolve_paths_by_folder(base_dir):
    """Returns (filepaths, labels) for a directory laid out as base_dir/<ClassName>/*.*"""
    paths, labels = [], []
    for class_name in cfg.CLASS_NAMES:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in glob.glob(os.path.join(class_dir, "*")):
            if os.path.isfile(f):
                paths.append(f)
                labels.append(cfg.CLASS_TO_IDX[class_name])
    return paths, labels


def _has_presplit_class_folders(split: str) -> bool:
    split_dir = os.path.join(cfg.IMAGES_DIR, split)
    if not os.path.isdir(split_dir):
        return False
    return any(os.path.isdir(os.path.join(split_dir, c)) for c in cfg.CLASS_NAMES)


def get_filepaths_and_labels(split: str):
    """
    split: 'train' or 'test'. Returns (filepaths list, integer labels list).
    Prints a summary so missing-data issues are obvious before training starts.
    Layout is auto-detected — see module docstring for the three supported layouts.
    """
    if _has_presplit_class_folders(split):
        # Layout 1: data/images/train/<Class>/*.jpg and data/images/test/<Class>/*.jpg
        # This is what the Kaggle "oasis-alzheimers-detection-multi-class-dataset" download
        # gives you directly — no reorganizing needed.
        split_dir = os.path.join(cfg.IMAGES_DIR, split)
        paths, labels = _resolve_paths_by_folder(split_dir)
        print(f"[{split}] {len(paths)} images found under {split_dir} (pre-split class folders).")

    elif cfg.MATCH_BY_ID:
        # Layout 2: data/images/<ID>.jpg, label from CSV
        csv_path = cfg.TRAIN_CSV if split == "train" else cfg.TEST_CSV
        df = _load_metadata(csv_path)
        paths, labels, missing = _resolve_paths_by_id(df)
        print(f"[{split}] {len(paths)} images found, {missing} CSV rows had no matching image file.")

    else:
        # Layout 3: data/images/<Class>/*.jpg, no existing split -> create a deterministic one
        all_paths, all_labels = _resolve_paths_by_folder(cfg.IMAGES_DIR)
        rng = np.random.RandomState(cfg.SEED)
        idx = rng.permutation(len(all_paths))
        cut = int(len(idx) * (1 - 0.2))
        chosen = idx[:cut] if split == "train" else idx[cut:]
        paths = [all_paths[i] for i in chosen]
        labels = [all_labels[i] for i in chosen]
        print(f"[{split}] {len(paths)} images found (auto-split from flat class folders).")

    if len(paths) == 0:
        print(f"[!] No images found for split='{split}'. "
              f"Have you placed images under {cfg.IMAGES_DIR}? See data/README_DATA.md.")

    if cfg.MAX_SAMPLES_PER_SPLIT is not None and len(paths) > cfg.MAX_SAMPLES_PER_SPLIT:
        rng = np.random.RandomState(cfg.SEED)
        idx = rng.choice(len(paths), size=cfg.MAX_SAMPLES_PER_SPLIT, replace=False)
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx]
        print(f"[{split}] capped to {len(paths)} images (config.MAX_SAMPLES_PER_SPLIT).")

    return paths, labels


def _make_dataset(paths, labels, training: bool, batch_size: int = None):
    batch_size = batch_size or cfg.BATCH_SIZE
    labels_onehot = tf.one_hot(labels, cfg.NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_onehot))

    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        if training:
            img = augment_image(img)
        return img, label

    if training:
        ds = ds.shuffle(buffer_size=max(len(paths), 1), seed=cfg.SEED)
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_train_val_test_datasets(batch_size: int = None):
    """
    Main entry point used by notebooks/train.py.
    Returns (train_ds, val_ds, test_ds, class_weights_dict).
    Validation is carved out of the training CSV (stratified) per config.VAL_SPLIT.
    """
    train_paths, train_labels = get_filepaths_and_labels("train")
    test_paths, test_labels = get_filepaths_and_labels("test")

    if len(train_paths) == 0:
        raise RuntimeError(
            "No training images found. Place MRI images under data/images/ "
            "per data/README_DATA.md before running this."
        )

    # Stratified train/val split
    from sklearn.model_selection import train_test_split
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=cfg.VAL_SPLIT,
        stratify=train_labels, random_state=cfg.SEED,
    )

    train_ds = _make_dataset(tr_paths, tr_labels, training=True, batch_size=batch_size)
    val_ds = _make_dataset(val_paths, val_labels, training=False, batch_size=batch_size)
    test_ds = _make_dataset(test_paths, test_labels, training=False, batch_size=batch_size)

    # Class weights to counter OASIS's class imbalance (NonDemented dominates)
    from collections import Counter
    counts = Counter(tr_labels)
    total = sum(counts.values())
    class_weights = {i: total / (cfg.NUM_CLASSES * counts.get(i, 1)) for i in range(cfg.NUM_CLASSES)}

    return train_ds, val_ds, test_ds, class_weights


def get_unlabeled_dataset_for_ssl(batch_size: int = None):
    """
    SSL pretraining just needs images, no labels — reuse the training split's images.
    """
    batch_size = batch_size or cfg.SSL_BATCH_SIZE
    train_paths, _ = get_filepaths_and_labels("train")
    ds = tf.data.Dataset.from_tensor_slices(train_paths)
    ds = ds.shuffle(buffer_size=max(len(train_paths), 1), seed=cfg.SEED)
    ds = ds.map(lambda p: load_and_preprocess_image(p), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
