# Bank Customer Churn Prediction

A Flask web app that predicts whether a bank customer is likely to churn,
using a Gradient Boosting classifier trained on the `Bank Customer Churn
Prediction.csv` dataset.

## Why ranking metrics, not just accuracy

The dataset's churn rate is ~20.4%, so a model that always predicts "no
churn" already scores ~79.6% accuracy without learning anything. To avoid
being misled by that baseline, evaluation reports ROC-AUC and PR-AUC
(threshold-independent ranking metrics) alongside accuracy/precision/
recall/F1.

## Decision threshold tuning

Classifying at the default 0.5 probability cutoff is arbitrary and not
tuned for this class balance. Instead:

1. 10-fold stratified cross-validation produces out-of-fold (OOF)
   predicted probabilities for every row — each prediction comes from a
   model that never saw that row during training.
2. The threshold that maximizes F1 on those OOF probabilities is selected
   (currently ~0.53, vs. the default 0.5).
3. That threshold is saved to `churn_metadata.pkl` and loaded by `app.py`
   at serving time, so the app's predictions match what was evaluated.

| Metric    | @ 0.5 threshold | @ tuned threshold |
|-----------|-----------------|--------------------|
| Accuracy  | 0.838           | 0.846              |
| Precision | 0.591           | 0.620              |
| Recall    | 0.654           | 0.630              |
| F1        | 0.621           | 0.625              |

10-fold CV ranking metrics (mean ± std): ROC-AUC 0.864 ± 0.018, PR-AUC
0.698 ± 0.033.

## Project structure

- `bank_customer_churn_model.py` — trains the model with stratified
  k-fold CV + SMOTE oversampling (applied only inside each training
  fold), computes OOF metrics, tunes the decision threshold, then
  refits a final model on the full dataset.
- `app.py` — Flask app that loads the trained model, scaler, training
  columns, and metadata (tuned threshold), and serves predictions.
- `gradient_boosting_model.pkl`, `scaler.pkl`, `train_columns.pkl`,
  `churn_metadata.pkl` — artifacts produced by the training script.
- `templates/`, `static/` — the web UI.

## Setup

```bash
pip install -r requirements.txt
```

## Retraining the model

```bash
python bank_customer_churn_model.py
```

This regenerates all four `.pkl` files from `Bank Customer Churn
Prediction.csv`.

## Running the app

```bash
python app.py
```

Then open http://127.0.0.1:5000 in a browser.
