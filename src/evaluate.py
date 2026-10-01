import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

from config import DEVICE


def evaluate(model, test_loader, threshold=0.5):
    """
    Print accuracy, AUC and a report per class for the model on the test set.
    threshold: probability from which a sample is classified as 1 (diseased).
    """

    model.to(DEVICE)
    model.eval()
    probs, targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu())
            targets.append(yb)

    probs = torch.cat(probs).numpy().ravel()
    targets = torch.cat(targets).numpy().ravel()
    predictions = probs >= threshold

    print(f"Test Accuracy: {accuracy_score(targets, predictions):.4f}")
    print(f"Test AUC: {roc_auc_score(targets, probs):.3f}")
    print(classification_report(targets, predictions, target_names=["healthy", "diseased"]))
