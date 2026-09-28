# Explainable Alzheimer's Disease Detection Using Deep Learning

An explainable deep learning framework for **multi-class Alzheimer's Disease detection from brain MRI images**, based on the architecture proposed by Tripathy et al. (2024), with additional **CBAM attention, Transformer-based global context, self-supervised learning, and explainability techniques**.

---

## 📌 Project Overview

This project reproduces and extends the architecture proposed in:

> **Tripathy et al. (2024)** — *Alzheimer's Disease Detection via Multiscale Feature Modelling Using Improved Spatial Attention Guided Depth Separable CNN*

**DOI:** 10.1007/s44196-024-00502-y

The original architecture uses a **Depthwise-Separable CNN with Improved Spatial Attention and Multiscale Feature Modelling** for Alzheimer's Disease classification.

Our work extends the baseline architecture with:

* **CBAM (Convolutional Block Attention Module)**
* **Transformer-based global-context branch**
* **Self-Supervised Learning (SSL)**
* **Explainable AI (XAI)** using Grad-CAM, SHAP, and LIME
* **Ablation study** to evaluate the contribution of each component

---

# 🧠 Contributions

The proposed framework builds progressively on the Tripathy et al. baseline:

```text
Tripathy Baseline
       │
       ▼
     + CBAM
       │
       ▼
+ Transformer Branch
       │
       ▼
+ Self-Supervised Learning
       │
       ▼
 Explainability (XAI)
```

### Our contributions

1. **CBAM Attention**

   * Adds channel and spatial attention mechanisms.
   * Helps the network focus on diagnostically relevant MRI regions.

2. **Transformer Global-Context Branch**

   * Captures long-range spatial relationships that may not be fully represented by convolutional layers.
   * Provides complementary global contextual information.

3. **Self-Supervised Pretraining**

   * Uses SimCLR-style contrastive learning before supervised classification.
   * Designed to improve feature representations without requiring labels during pretraining.

4. **Explainable AI**

   * **Grad-CAM** for visualizing important regions.
   * **SHAP** for feature-level contribution analysis.
   * **LIME** for local prediction explanations.

5. **Ablation Study**

   * Experiments progressively add proposed components.
   * Performance is evaluated using Accuracy, Macro-F1, and ROC-AUC.

---

# 📊 Results

The ablation study was performed using the **OASIS Alzheimer's MRI dataset**.

The baseline, CBAM, and CBAM + Transformer models were trained using the full dataset with early stopping.

The SSL experiment was performed at reduced scale because of computational constraints.

| Experiment                 | Accuracy | Macro F1 | ROC-AUC | Scale        |
| -------------------------- | -------: | -------: | ------: | ------------ |
| Tripathy Baseline          |    0.600 |    0.271 |   0.817 | Full dataset |
| + CBAM                     |    0.818 |    0.316 |   0.867 | Full dataset |
| + CBAM + Transformer       |    0.765 |    0.337 |   0.841 | Full dataset |
| + CBAM + Transformer + SSL |    0.019 |    0.022 |   0.636 | Reduced      |

### 🔎 Key Finding

Accuracy alone does not fully represent model performance on this dataset.

For example, the **CBAM model achieves 81.8% accuracy but only 0.316 Macro-F1**, indicating that the model is strongly influenced by the majority class.

The OASIS dataset has substantial class imbalance, particularly affecting the minority classes:

* `MildDemented`
* `ModerateDemented`

The Macro-F1 score increases across the first three experiments:

```text
Baseline                 → 0.271
Baseline + CBAM          → 0.316
Baseline + CBAM + Trans. → 0.337
```

This suggests that the additional attention and global-context mechanisms improve performance across classes, particularly in terms of the balance between precision and recall.

---

## ⚠️ SSL Experiment Limitation

The SSL-pretrained model achieved:

```text
Accuracy : 1.9%
Macro F1 : 2.2%
ROC-AUC  : 0.636
```

However, this experiment was conducted using only **2,000 images**, with:

* 3 SSL pretraining epochs
* 8 supervised training epochs

The reduced-scale experiment used a random, non-stratified subsampling procedure. At this small sample size, the resulting subset can become severely imbalanced or potentially contain very few samples from some classes.

Therefore, the SSL result **should not be interpreted as evidence that self-supervised learning reduces performance**.

A full-scale experiment using the complete dataset and longer SSL pretraining is required for a fair comparison.

### Future SSL experiment

A more reliable evaluation would use:

* Full training dataset
* Stratified sampling where applicable
* Approximately 30 SSL pretraining epochs
* Full supervised training schedule
* Multiple random seeds

---

# 🗂️ Dataset

### OASIS Alzheimer's Detection Multi-Class Dataset

**Kaggle:**
https://www.kaggle.com/datasets/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset

The project uses four classes:

```text
NonDemented
VeryMildDemented
MildDemented
ModerateDemented
```

These classes correspond to the four-way classification setting used by the baseline paper.

### Dataset Size

The Kaggle release contains approximately:

```text
Training images : ~55,000
Testing images  : ~46,000
```

This is substantially larger than the original OASIS-1 patient population.

The Kaggle release is a **Roboflow-exported and augmented dataset**, meaning that multiple augmented images may originate from the same original MRI scan.

### ⚠️ Dataset Leakage Consideration

An exact filename-overlap check between training and testing sets returned:

```text
Exact duplicate filenames: 0
```

However, filename-level checking cannot rule out **near-duplicate augmented images derived from the same original scan**.

Therefore, patient-level or source-image-level separation should be considered when performing a rigorous future evaluation.

---

# 🏗️ Project Architecture

```text
                         MRI Image
                            │
                            ▼
                    Preprocessing
                            │
                            ▼
                  Data Augmentation
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
    Depthwise-Separable CNN        Transformer Branch
             │                             │
             ▼                             ▼
      Improved Spatial             Global Context
      Attention Block                 Features
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                         CBAM
                            │
                            ▼
                       Feature Fusion
                            │
                            ▼
                      Classification
                            │
                            ▼
                    Alzheimer's Class
                            │
                            ▼
                  Explainability Layer
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              Grad-CAM    SHAP      LIME
```

---

# 📁 Project Structure

```text
alz_project/
│
├── data/
│   ├── oasis_train_patients_metadata.csv
│   ├── oasis_test_patients_metadata.csv
│   ├── README_DATA.md
│   └── images/
│       └── (not tracked in Git)
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   │
│   ├── models/
│   │   ├── isab.py
│   │   ├── depthwise_cnn.py
│   │   ├── cbam.py
│   │   ├── transformer_branch.py
│   │   ├── fusion_model.py
│   │   └── ssl_pretrain.py
│   │
│   └── explainability/
│       ├── gradcam.py
│       ├── shap_explain.py
│       └── lime_explain.py
│
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_cbam_experiment.ipynb
│   ├── 05_transformer_experiment.ipynb
│   ├── 06_ssl_pretraining.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_ablation_comparison.ipynb
│
├── tests/
│   └── smoke_test.py
│
├── results/
│   ├── baseline_metrics.json
│   ├── cbam_metrics.json
│   ├── cbam_transformer_metrics.json
│   ├── cbam_transformer_ssl_metrics.json
│   └── ablation_table.csv
│
└── requirements.txt
```

---

# ⚙️ Setup

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace `YOUR_USERNAME/YOUR_REPOSITORY` with your GitHub repository name.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🗃️ Dataset Setup

The actual MRI image files are **not included in this repository** because of their large size.

Follow:

```text
data/README_DATA.md
```

for instructions on downloading and placing the dataset.

The expected structure is:

```text
data/
├── oasis_train_patients_metadata.csv
├── oasis_test_patients_metadata.csv
└── images/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
```

---

# 🚀 Running the Project

## Environment Setup

Open:

```text
notebooks/01_environment_setup.ipynb
```

and verify that the required libraries and TensorFlow environment are correctly configured.

---

## 📊 Data Exploration

Run:

```text
notebooks/02_data_exploration.ipynb
```

This notebook explores:

* Dataset distribution
* Class imbalance
* Image dimensions
* Metadata
* Sample MRI images

---

# 🧪 Ablation Experiments

The project contains four primary model experiments.

### Experiment 1 — Tripathy Baseline

```bash
python -m src.train --experiment baseline
```

Reproduces the baseline architecture from Tripathy et al.

---

### Experiment 2 — CBAM

```bash
python -m src.train --experiment cbam
```

Adds the **Convolutional Block Attention Module**.

---

### Experiment 3 — CBAM + Transformer

```bash
python -m src.train --experiment cbam_transformer
```

Adds a Transformer-based global-context branch alongside the CNN architecture.

---

### Experiment 4 — CBAM + Transformer + SSL

```bash
python -m src.train --experiment cbam_transformer_ssl
```

Uses self-supervised pretraining followed by supervised classification.

> The published result for this experiment was obtained using a reduced dataset and shortened training schedule because of computational constraints.

---

# 📈 Evaluation

After completing the training experiments, run:

```bash
python -m src.evaluate --ablation
```

This generates the final comparison between the experiments.

Alternatively, run:

```text
notebooks/08_ablation_comparison.ipynb
```

The notebook reads the saved metric files from:

```text
results/
```

and generates the ablation comparison table.

---

# 🔍 Explainability

The project includes three explainability methods.

## Grad-CAM

**Gradient-weighted Class Activation Mapping** is used to visualize the image regions contributing most strongly to a model prediction.

```text
MRI Image
   ↓
CNN Feature Maps
   ↓
Gradients
   ↓
Importance Weights
   ↓
Heatmap
```

---

## SHAP

**SHAP (SHapley Additive exPlanations)** is used to estimate the contribution of input features to individual predictions.

Implementation:

```text
src/explainability/shap_explain.py
```

---

## LIME

**Local Interpretable Model-Agnostic Explanations** provides local explanations by perturbing the input and observing how predictions change.

Implementation:

```text
src/explainability/lime_explain.py
```

---

# 🧪 Explainability Notebook

Run:

```text
notebooks/07_explainability.ipynb
```

to generate explanations for selected MRI predictions.

The goal is not only to determine **which class the model predicts**, but also to investigate **which image regions contribute to that prediction**.

---

# 🏷️ Attribution & Reproducibility

Each model implementation contains header comments identifying the origin of the corresponding components.

### `[Tripathy paper]`

Direct reproduction of components from the baseline architecture.

### `[our contribution]`

Components introduced in this project:

* CBAM
* Transformer branch
* SSL
* Grad-CAM
* SHAP
* LIME

### `[reference]`

Implementation patterns or structural ideas adapted from publicly available repositories.

These references were consulted for implementation ideas and project structure and were **not copied wholesale**.

---

# 📚 References

### Base Paper

Tripathy et al. (2024).

**Alzheimer's Disease Detection via Multiscale Feature Modelling Using Improved Spatial Attention Guided Depth Separable CNN.**

*International Journal of Computational Intelligence Systems.*

DOI:

```text
10.1007/s44196-024-00502-y
```

---

### Reference Repositories

* https://github.com/matthewchung74/hack4health_alzheimers
* https://github.com/carloalbertobarbano/contrastive-learning-brain-mri
* https://github.com/rkushol/ADDFormer
* https://github.com/thanhkaist/AttentionResnet

---

# 📌 Limitations

The current implementation has several limitations:

1. **Class imbalance**

   * The dataset contains substantially different numbers of samples across diagnostic classes.
   * Accuracy therefore does not fully capture minority-class performance.

2. **Augmented dataset**

   * The Kaggle dataset contains many augmented images derived from a relatively small original patient population.

3. **Potential near-duplicate leakage**

   * Exact filename duplication was not found, but augmented copies of the same original MRI may potentially occur across training and testing sets.

4. **Reduced-scale SSL experiment**

   * The SSL experiment used only 2,000 images and a shortened training schedule.
   * Therefore, its results are not directly comparable with the full-scale experiments.

5. **Computational requirements**

   * Full-scale training is computationally expensive and benefits significantly from GPU acceleration.

---

# 🔮 Future Work

Future improvements include:

* Full-scale SSL pretraining using the entire dataset
* Stratified sampling during reduced-scale experiments
* Patient-level train/test splitting
* Source-image-level duplicate detection
* Class-weighted or focal-loss training
* More extensive hyperparameter tuning
* Multiple random-seed evaluations
* Additional Transformer architectures
* Stronger self-supervised learning strategies
* Quantitative evaluation of explainability methods
* External validation using an independent MRI dataset
* Clinical expert evaluation of generated explanations

---

# 💻 Hardware Requirements

Full-scale training on approximately 55,000 training images requires GPU acceleration.

A **Google Colab T4 GPU** is sufficient for the current codebase, although training time depends on:

* Image resolution
* Batch size
* Number of epochs
* Model architecture
* Data augmentation
* Storage and data-loading speed

CPU-only training can be significantly slower and may take approximately **10 hours per epoch** depending on the hardware and configuration.

---

# 📜 License

This project is intended for **academic and research purposes**.

The original architecture is based on the work of Tripathy et al. (2024), and external repositories were consulted for implementation patterns and ideas. Please refer to the respective original sources for their licensing and attribution requirements.

---

# ⚠️ Disclaimer

This project is a **research and educational implementation** and is not intended to provide medical diagnoses or replace evaluation by qualified healthcare professionals.

Model predictions and visual explanations should not be interpreted as clinical diagnoses.

---

# 👩‍💻 Project Summary

**Project:** Explainable Alzheimer's Disease Detection Using Deep Learning

**Base Architecture:** Improved Spatial Attention Guided Depthwise-Separable CNN

**Dataset:** OASIS-1 MRI — Kaggle augmented release

**Classification:** 4-class Alzheimer's Disease detection

**Additional Components:**

```text
CBAM
Transformer Global-Context Branch
Self-Supervised Learning
Grad-CAM
SHAP
LIME
```

**Evaluation:** Accuracy, Macro-F1, ROC-AUC, Confusion Matrix, Ablation Study

**Framework:** TensorFlow / Keras

**Environment:** Python, Jupyter, VS Code, Google Colab

---

⭐ If you find this project useful for research or learning, consider starring the repository.
