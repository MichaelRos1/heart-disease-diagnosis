import copy
import torch
import torch.nn as nn

from config import LEARNING_RATE, EPOCHS, DEVICE, PATIENCE


def train_epoch(model, loader, criterion, optimizer):
    """Run one pass over the training data and return average loss per sample."""

    model.train()
    total_loss = 0.0
    n=0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        # Weighted by batch size, because of a smaller last batch
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss/n


def validate(model, loader, criterion):
    """Run the model on the validation data and return the average loss. (no weight updates)"""

    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits,yb)

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

    return total_loss / n
 

def fit(model, train_loader, val_loader):
    """Train the model with early stopping and return it with the best weights."""

    print(f"Training, using the Device: {DEVICE}")

    model.to(DEVICE)
    y_train = train_loader.dataset.tensors[1]
    # pos_weight, so the rarer class is weighted more
    pos_weight = ((y_train == 0).sum() / (y_train == 1).sum()).reshape(1).to(DEVICE) 

    criterion_training = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Unweighted loss, to judge the model on the plain loss
    criterion_validation = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion_training, optimizer)
        val_loss = validate(model, val_loader, criterion_validation)

        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # deepcopy to protect against changes in state_dict
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement for {PATIENCE:d} epochs. Stopping.")
                break
    model.load_state_dict(best_weights)
    return model
