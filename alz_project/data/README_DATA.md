# Getting the actual MRI images

The CSVs in this folder (`oasis_train_patients_metadata.csv`, `oasis_test_patients_metadata.csv`)
only contain patient metadata (ID, age, CDR, class label, etc.) — no pixels. You need the image
files themselves from Kaggle.

## 1. Download

https://www.kaggle.com/datasets/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset

## 2. Expected layout (this is what the Kaggle zip already gives you — no reorganizing needed)

This dataset comes pre-split into `train/` and `test/`, each containing one folder per class.
Just place that structure under `data/images/` in this project:

```
data/images/
├── train/
│   ├── NonDemented/        *.jpg
│   ├── VeryMildDemented/   *.jpg
│   ├── MildDemented/       *.jpg
│   └── ModerateDemented/   *.jpg
└── test/
    ├── NonDemented/        *.jpg
    ├── VeryMildDemented/   *.jpg
    ├── MildDemented/       *.jpg
    └── ModerateDemented/   *.jpg
```

`src/data_loader.py` auto-detects this layout and uses it directly — labels come from the folder
names, and this preserves the dataset's original train/test split (matching the metadata CSVs)
rather than creating a new random split.

Two other layouts are supported as fallbacks (see the docstring at the top of
`src/data_loader.py`) if your download ever looks different: a flat `<ID>.jpg` layout keyed to
the CSV, or a single flat class-folder layout with no existing split.

## 3. Verify

Run `notebooks/02_data_exploration.ipynb` — the image-availability cell counts images found per
class/split so you know before training whether the pipeline can see your data.
