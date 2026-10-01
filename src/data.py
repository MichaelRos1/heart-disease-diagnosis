import torch
from torch.utils.data import TensorDataset, DataLoader
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import KAGGLE_DATASET, DATA_DIR, CSV_PATH, NUMERIC_COLS, CATEGORICAL_COLS, BATCH_SIZE



def download_data_if_needed():
    """ Download dataset via kagglehub on the first run and cache it in the folder data/. """

    if CSV_PATH.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    cache_dir = Path(kagglehub.dataset_download (KAGGLE_DATASET))

    csv_files = list(cache_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV found")

    shutil.copy(csv_files[0], CSV_PATH)


def load_data():
    "Load the raw dataset and return it as a DataFrame"

    download_data_if_needed()
    return pd.read_csv(CSV_PATH, sep=";")


def impute(df):
    """
    Fill missing values and split the DataFrame into features and target.
    Numerical features are filled with the median.
    Categorical features are filled with the mode.
    """

    # No information gained from "id", therefore exclude it from the dataset
    df = df.drop(columns=["id"])

    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    return X,y


def split(X: pd.DataFrame, y: pd.DataFrame):
    """Split into 70% train, 15% validation, 15% test (stratified)"""

    X_train,X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    X_val,X_test, y_val,y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )
    return X_train,X_test,X_val, y_train,y_test,y_val


def scale(X_train,X_test,X_val):
    """Scale numeric features to mean 0 and std 1."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_val = X_val.copy()

    scaler = StandardScaler()
    # Fit only on the training set to avoid data leakage
    scaler.fit(X_train[NUMERIC_COLS])

    X_train[NUMERIC_COLS] = scaler.transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])

    return X_train, X_test, X_val, scaler


def to_tensor_dataset(X, y):
    """Convert features and labels into PyTorch Tensor Dataset."""
    X_t = torch.tensor(np.asarray(X, dtype="float32"))
    y_t = torch.tensor(np.asarray(y, dtype="float32")).reshape(-1, 1)
    return TensorDataset(X_t, y_t)


def make_loaders(X_train,X_test,X_val, y_train,y_test,y_val):
    """Wrap the splits into DataLoaders."""
    train_ds = to_tensor_dataset(X_train, y_train)
    val_ds = to_tensor_dataset(X_val, y_val)
    test_ds = to_tensor_dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader,val_loader,test_loader


def preprocessing(df):
    """Run the pipeline: input data -> impute -> split -> scale -> wrap in DataLoaders"""
    
    X,y = impute(df)
    X_train,X_test,X_val, y_train,y_test,y_val = split(X,y)
    X_train,X_test,X_val, scaler = scale(X_train,X_test,X_val)

    return make_loaders(X_train,X_test,X_val, y_train,y_test,y_val), scaler