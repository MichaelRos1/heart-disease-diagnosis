# Heart Disease Diagnosis

A small PyTorch project that predicts whether a patient has a cardiovascular disease from basic health and lifestyle data.

This is a learning project, not a medical tool!


## Results

Evaluated on a unbiased test set of 10500 patients.

| Model               | Test AUC |
|---------------------|----------|
| Neural network      | 0.796    |
| Logistic regression | 0.773    |
| Random forest       | 0.772    |

At a decision threshold of 0.5 the neural network has 0.73 accuracy.
The AUC is measured on the predicted probabilities and does not depend on the threshold.

Exact numbers can vary slightly.

The notebook `notebooks/model_analysis.ipynb`covers a more in depth and hands on analysis.

## Dataset

[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
70 000 patient records with 11 features and the target `cardio`
(1 = cardiovascular disease, 0 = healthy).
Balanced classes.

The data is downloaded and stored automatically on the first run.

| Feature | Meaning |
|---------|---------|
| `age` | Age in days |
| `gender` | 1 = female, 2 = male |
| `height`, `weight` | in cm and kg |
| `ap_hi`, `ap_lo` | Systolic and diastolic blood pressure |
| `cholesterol`, `gluc` | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke`, `alco`, `active` | 0 = no, 1 = yes |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The model is the class `HDD` in `src/model.py`.
Load and prepare the data, create an `HDD` object and use its methods.

```Python
from config import DROPOUT, INPUT_SIZE
from data import load_data, preprocessing

# 1. Prepare the data
df = load_data()
(train_loader, val_loader, test_loader), scaler = preprocessing(df)

# 2. Create the model
model = HDD(input_size=INPUT_SIZE, dropout=DROPOUT)

# 3. Use its methods
model = model.fit(train_loader, val_loader) # training with early stopping
model.evaluate(test_loader)                 # accuracy, AUC and classification report
```

| Method | Description |
|--------|-------------|
| `fit(train_loader, val_loader)` | Trains the model with early stopping and restores the weights of the best epoch. Returns the model. |
| `evaluate(test_loader, threshold=0.5)` | Prints accuracy, AUC and a per-class report. A patient counts as diseased, if the probability passes the threshold. |
| `predict(patient, scaler, threshold=0.5)` | Classifies a single patient. Returns the probability of disease and the decision (1 = diseased). |

For a practical example of how a new prediction could look like, please refer to the notebook `notebooks/patient_prediction.ipynb`.

## Inner Workings

- **Split:** 70% train, 15% validation, 15% test, stratified.
- **Preprocessing:** fill missing numeric values with the median and missing categorical values with the mode.
- **Model:** feed-forward network 32 -> 16 -> 1, ReLU activations and dropout layers.
- **Training:** Adam learner with learning rate 1e-3, batch size 32, 100 epochs maximum. Early stopping after 10 epochs without change. Restore the weights of the best epoch. (Parameters can be changed in `src/config.py`)
- **Class weighting:** `BCEWithLogitsLoss* uses a `pos_weight`. On this balanced set it is close to 1. It matters for data with imbalanced classes.
