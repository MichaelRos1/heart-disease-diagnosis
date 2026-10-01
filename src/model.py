import torch
import torch.nn as nn
import pandas as pd

from config import FEATURE_COLS, NUMERIC_COLS
from train import fit as fit_model
from evaluate import evaluate as evaluate_model

class HDD(nn.Module):
    """
    Small network for binary classification. (heart disease: yes/no)
    Architecture: input -> 32 -> 16 -> 1, ReLU and dropout after each hidden layer.
    """
    def __init__(self, input_size, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16,1)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, patient, scaler, threshold=0.5):
        """ 
        patient: dict with one value per feature (see dataset description)
        scaler: the StandardScaler fitted on the training data

        Return: probability of disease, classification depending on threshold
        """
        df = pd.DataFrame([patient])[FEATURE_COLS].astype("float64")
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
        x = torch.tensor(df.values, dtype=torch.float32, device=next(self.parameters()).device)

        self.eval()
        with torch.no_grad():
            prob = torch.sigmoid(self(x)).item()
        return prob, prob >= threshold


    # Wrappers so the model can be used more conveniently
    def fit(self, train_loader, val_loader):
        return fit_model(self, train_loader, val_loader)

    def evaluate(self, test_loader, threshold=0.5):
        return evaluate_model(self, test_loader, threshold)



