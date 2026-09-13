# Explainable Alzheimer's Disease Detection Using Deep Learning

Baseline: Tripathy et al. (2024), *"Alzheimer's Disease Detection via Multiscale Feature
Modelling Using Improved Spatial Attention Guided Depth Separable CNN,"* Int. J. Computational
Intelligence Systems. DOI: 10.1007/s44196-024-00502-y

Our contributions on top of the baseline: **CBAM**, a **Transformer global-context branch**,
**self-supervised pretraining (SSL)**, and **explainability (Grad-CAM, SHAP, LIME)**, evaluated
via an ablation study (Experiments 1–5).

---

## Status — what's verified vs. what's still pending

This repo is a **complete, runnable codebase**, and the full pipeline (data loading, all four
model architectures, training loop, metrics, checkpointing) has been **verified end-to-end on
the real OASIS dataset** (Kaggle: shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset,
~55k train / ~46k test images after Roboflow augmentation) — a 200-image-capped, 3-epoch run
completed cleanly with train accuracy rising 25.3% → 36.5% and loss falling 1.59 → 1.38,
confirming real images load correctly, labels align with folders, and the model learns.

What's still pending, honestly:
- **Full-scale results.** The verified run above was intentionally capped
  (`config.MAX_SAMPLES_PER_SPLIT`) to a few minutes for validation speed. The real 40-epoch,
  full-dataset, 4-experiment ablation study has not been completed yet — it requires a GPU
  (CPU-only, one epoch on the full ~55k-image training set takes on the order of 10 hours; see
  the note on dataset size below).
- **Train/test leakage check.** This Kaggle release is a Roboflow-augmented export — the same
  original scan can appear as multiple augmented copies, potentially split across both train and
  test folders. This hasn't been audited yet and should be addressed (or at least disclosed)
  before citing final numbers, since it can inflate test accuracy if present.

Once the full ablation completes (recommended: Google Colab's free T4 GPU — this exact code runs
there unchanged), replace this section with the real accuracy/F1/AUC numbers from
`results/ablation_table.csv`.

## Dataset

Kaggle: "OASIS Alzheimer's Detection Multi-Class Dataset (MRI Images)", source OASIS-1
https://www.kaggle.com/datasets/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset

Classes (4-way, matching the base paper): `NonDemented`, `VeryMildDemented`, `MildDemented`,
`ModerateDemented`. Note: this Kaggle release is a Roboflow-exported, augmented version (~55k
train / ~46k test images) — much larger than the original OASIS-1 patient count (~400 scans),
because each original scan was expanded into multiple augmented copies before export.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then follow `data/README_DATA.md` to place images, and open `notebooks/01_environment_setup.ipynb`
in VS Code / Jupyter.

## Project layout

```
alz_project/
├── data/
│   ├── oasis_train_patients_metadata.csv   (provided by you)
│   ├── oasis_test_patients_metadata.csv    (provided by you)
│   ├── README_DATA.md                      (how to get + place the actual images)
│   └── images/                             (you create this — one file per patient ID)
├── src/
│   ├── config.py            Central config (image size, classes, paths, hyperparams)
│   ├── data_loader.py       CSV -> tf.data.Dataset pipeline
│   ├── preprocessing.py     Resize / normalize / (optional) skull-strip hook
│   ├── augmentation.py      Rotation, flip, crop, contrast jitter
│   ├── utils.py             Seeding, plotting, metrics helpers
│   ├── train.py             CLI trainer for any experiment variant
│   ├── evaluate.py          Metrics + confusion matrix + ablation table builder
│   ├── models/
│   │   ├── isab.py                 Improved Spatial Attention Block  [Tripathy paper]
│   │   ├── depthwise_cnn.py        10-layer depthwise-separable CNN backbone + I-SAB + fusion + MLP  [Tripathy paper]
│   │   ├── cbam.py                 Channel + spatial attention  [our contribution]
│   │   ├── transformer_branch.py   ViT-style global-context branch  [our contribution]
│   │   ├── fusion_model.py         Builds Experiments 1-4 by composing the above  [our contribution]
│   │   └── ssl_pretrain.py         SimCLR-style contrastive + masked-patch pretraining  [our contribution]
│   └── explainability/
│       ├── gradcam.py       [our contribution]
│       ├── shap_explain.py  [our contribution]
│       └── lime_explain.py  [our contribution]
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_baseline_model.ipynb          Experiment 1: Tripathy baseline
│   ├── 04_cbam_experiment.ipynb         Experiment 2: + CBAM
│   ├── 05_transformer_experiment.ipynb  Experiment 3: + CBAM + Transformer
│   ├── 06_ssl_pretraining.ipynb         Experiment 4: + SSL pretraining
│   ├── 07_explainability.ipynb          Experiment 5: Grad-CAM + SHAP + LIME
│   └── 08_ablation_comparison.ipynb     Final comparison table across all 4 experiments
├── tests/
│   └── smoke_test.py        Runs every model on random dummy tensors to verify shapes
├── results/                 Metrics/plots land here after you run notebooks
└── requirements.txt
```

## Attribution key

Every model file has a header comment marking each block as one of:
- **[Tripathy paper]** — direct reproduction of the base architecture
- **[our contribution]** — CBAM / Transformer / SSL / XAI additions
- **[reference]** — pattern borrowed/adapted from a public GitHub repo (cited inline), not copy-pasted verbatim

Reference repos consulted for structure/ideas (not copied wholesale):
- https://github.com/matthewchung74/hack4health_alzheimers
- https://github.com/carloalbertobarbano/contrastive-learning-brain-mri
- https://github.com/rkushol/ADDFormer
- https://github.com/thanhkaist/AttentionResnet

## Running the ablation

```bash
python src/train.py --experiment baseline
python src/train.py --experiment cbam
python src/train.py --experiment cbam_transformer
python src/train.py --experiment cbam_transformer_ssl
python src/evaluate.py --ablation
```

or just run `notebooks/08_ablation_comparison.ipynb` after the four training runs, which reads
each experiment's saved `results/<experiment>_metrics.json` and builds the comparison table.
