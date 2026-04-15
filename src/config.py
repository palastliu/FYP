from pathlib import Path

# Project root
PROJECT_ROOT = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL")

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

# Original dataset files
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Cleaned dataset files
TRAIN_CLEAN_FILE = PROCESSED_DATA_DIR / "train_clean.csv"
VAL_CLEAN_FILE = PROCESSED_DATA_DIR / "val_clean.csv"
TEST_CLEAN_FILE = PROCESSED_DATA_DIR / "test_clean.csv"

# Model save path
BERT_MODEL_DIR = MODELS_DIR / "bert_baseline"

# Label mapping
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Model settings
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
RANDOM_SEED = 42