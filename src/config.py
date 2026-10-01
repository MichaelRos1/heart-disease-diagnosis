from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KAGGLE_DATASET = "sulianova/cardiovascular-disease-dataset"
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "cardiovascular_disease.csv"

NUMERIC_COLS = ["age", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
FEATURE_COLS = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]

BATCH_SIZE = 32
DROPOUT = 0.2 # Fraction of activations that are dropped in training
INPUT_SIZE = len(FEATURE_COLS)

LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10 # Number of epochs without improvement before early stopping

# Select Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")