"""
Central configuration for the Explainable Alzheimer's Detection project.
[Tripathy paper]: IMG_SIZE, NUM_CLASSES, CLASS_NAMES follow the base paper's setup
(4-class OASIS classification). We start at 128x128 grayscale per the project plan
to keep early development/training fast; bump to 176x176 or 224x224 once the
pipeline is verified end-to-end, if your GPU/time budget allows.
"""
import os

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_CSV = os.path.join(DATA_DIR, "oasis_train_patients_metadata.csv")
TEST_CSV = os.path.join(DATA_DIR, "oasis_test_patients_metadata.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ---- Image / label config ---------------------------------------------------
IMG_SIZE = 128           # 128x128 grayscale to start (see README)
CHANNELS = 1
IMG_EXT = ".jpg"         # change if your downloaded images are .png
CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# If your images are in class-named folders rather than <ID>.jpg files,
# set this False and see data/README_DATA.md -> build_flat_image_index().
MATCH_BY_ID = True

# ---- Training hyperparameters ------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15          # carved out of the training CSV
SEED = 42

# Set to an integer (e.g. 500) to cap how many images per split are used — lets you run a
# fast end-to-end validation pass on real data (minutes, not hours) before committing to a
# full run. Set to None to use the full dataset.
MAX_SAMPLES_PER_SPLIT = None

# ---- SSL pretraining ---------------------------------------------------------
SSL_EPOCHS = 30
SSL_BATCH_SIZE = 64
SSL_PROJECTION_DIM = 128
SSL_TEMPERATURE = 0.1

# ---- Transformer branch -----------------------------------------------------
TRANSFORMER_PATCH_SIZE = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_EMBED_DIM = 128
TRANSFORMER_MLP_DIM = 256

# ---- Experiment names (used for filenames in results/) -----------------------
EXPERIMENTS = ["baseline", "cbam", "cbam_transformer", "cbam_transformer_ssl"]
