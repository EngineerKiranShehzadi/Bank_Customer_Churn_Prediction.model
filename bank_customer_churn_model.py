import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

import joblib

# ---------------------------
# 1) Load dataset
# ---------------------------
df = pd.read_csv("Bank Customer Churn Prediction.csv")

if "customer_id" in df.columns:
    df = df.drop("customer_id", axis=1)

# ---------------------------
# 2) Separate X and y
# ---------------------------
X = df.drop("churn", axis=1)
y = df["churn"]

# ---------------------------
# 3) Baseline accuracy / class balance
# ---------------------------
# At a ~20% churn rate, a model that always predicts "no churn" already
# scores ~80% accuracy, so accuracy alone can't tell a useful model from
# a useless one. Report the ranking metrics (ROC-AUC/PR-AUC) alongside it.
baseline_acc = y.value_counts(normalize=True).max()
churn_rate = y.value_counts(normalize=True).min()
print("Baseline Accuracy (majority class):", baseline_acc)
print("Churn rate (positive class):", churn_rate)

# ---------------------------
# 4) One-hot encode ONCE (for all data)
# ---------------------------
X = pd.get_dummies(X, drop_first=True)

# ---------------------------
# 5) Decide K so that:
#       - K <= 10
#       - each fold has at least 30 instances
# ---------------------------
n_samples = len(X)
max_splits_by_size = n_samples // 30

if max_splits_by_size < 2:
    raise ValueError(
        f"Not enough samples ({n_samples}) to create at least 2 folds "
        f"with >= 30 instances each."
    )

k = min(10, max_splits_by_size)
print(f"\nUsing StratifiedKFold with k = {k}")

# ---------------------------
# 6) K-Fold Cross-Validation
#    (scaling + SMOTE inside each fold)
#    Also collect out-of-fold (OOF) predicted probabilities so the
#    decision threshold can be tuned on held-out data, not the training
#    folds, and so serving-time behaviour matches evaluation.
# ---------------------------
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []
fold_roc_aucs = []
fold_pr_aucs = []

oof_proba = np.zeros(len(X))

fold_num = 1

for train_idx, test_idx in skf.split(X, y):
    print(f"\n===== Fold {fold_num} / {k} =====")

    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_fold)

    print("Class distribution after SMOTE (this fold):")
    print(pd.Series(y_train_res).value_counts())

    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train_res, y_train_res)

    y_proba_fold = gb_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_fold = (y_proba_fold >= 0.5).astype(int)

    oof_proba[test_idx] = y_proba_fold

    acc = accuracy_score(y_test_fold, y_pred_fold)
    prec = precision_score(y_test_fold, y_pred_fold, zero_division=0)
    rec = recall_score(y_test_fold, y_pred_fold, zero_division=0)
    f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)
    roc_auc = roc_auc_score(y_test_fold, y_proba_fold)
    pr_auc = average_precision_score(y_test_fold, y_proba_fold)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)
    fold_roc_aucs.append(roc_auc)
    fold_pr_aucs.append(pr_auc)

    fold_num += 1

# ---------------------------
# 7) Overall CV metrics (mean +/- std across folds)
# ---------------------------
cv_metrics = {
    "accuracy_mean": np.mean(fold_accuracies), "accuracy_std": np.std(fold_accuracies),
    "precision_mean": np.mean(fold_precisions), "precision_std": np.std(fold_precisions),
    "recall_mean": np.mean(fold_recalls), "recall_std": np.std(fold_recalls),
    "f1_mean": np.mean(fold_f1s), "f1_std": np.std(fold_f1s),
    "roc_auc_mean": np.mean(fold_roc_aucs), "roc_auc_std": np.std(fold_roc_aucs),
    "pr_auc_mean": np.mean(fold_pr_aucs), "pr_auc_std": np.std(fold_pr_aucs),
}

print("\n===== Cross-Validation Results (Mean +/- Std) =====")
for name in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
    print(f"{name:10s}: {cv_metrics[name + '_mean']:.4f} +/- {cv_metrics[name + '_std']:.4f}")

# ---------------------------
# 8) Tune the decision threshold on out-of-fold predictions
#    (never on the training folds, and never on the final refit) and
#    report accuracy/precision/recall/f1 at the default 0.5 cutoff vs.
#    the tuned cutoff so the gain is visible.
# ---------------------------
def metrics_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

oof_metrics_default = metrics_at_threshold(y, oof_proba, 0.5)

candidate_thresholds = np.unique(oof_proba)
best_threshold = 0.5
best_f1 = -1.0
for t in candidate_thresholds:
    f1 = f1_score(y, (oof_proba >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

oof_metrics_tuned = metrics_at_threshold(y, oof_proba, best_threshold)

print(f"\nTuned decision threshold (max F1 on OOF predictions): {best_threshold:.4f}")
print("OOF metrics @ default 0.5 threshold:", oof_metrics_default)
print("OOF metrics @ tuned threshold      :", oof_metrics_tuned)

# ---------------------------
# 9) Train FINAL model on full dataset
#    (after seeing CV performance)
# ---------------------------
print("\nTraining final model on FULL data...")

final_scaler = StandardScaler()
X_scaled_full = final_scaler.fit_transform(X)

final_smote = SMOTE(random_state=42)
X_res_full, y_res_full = final_smote.fit_resample(X_scaled_full, y)

print("\nClass distribution after SMOTE on full data:")
print(pd.Series(y_res_full).value_counts())

final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X_res_full, y_res_full)

# ---------------------------
# 10) Save model + scaler + columns + metadata
#     The tuned threshold and evaluation metrics are persisted alongside
#     the model so app.py applies the same cutoff used to evaluate it,
#     instead of silently defaulting back to 0.5.
# ---------------------------
joblib.dump(final_model, "gradient_boosting_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "train_columns.pkl")

churn_metadata = {
    "threshold": float(best_threshold),
    "cv_metrics": cv_metrics,
    "oof_metrics_default": oof_metrics_default,
    "oof_metrics_tuned": oof_metrics_tuned,
    "churn_rate": float(round(churn_rate, 4)),
    "baseline_accuracy": float(round(baseline_acc, 4)),
    "k_folds": k,
}
joblib.dump(churn_metadata, "churn_metadata.pkl")

print("\nFinal model, scaler, columns, and metadata saved successfully.")

# ---------------------------
# 11) Example: Evaluate on a random sample of REAL (unresampled) data,
#     using the tuned threshold.
# ---------------------------
df_eval = pd.read_csv("Bank Customer Churn Prediction.csv")
if "customer_id" in df_eval.columns:
    df_eval = df_eval.drop("customer_id", axis=1)

sample = df_eval.sample(1000, random_state=1)

X_real = sample.drop("churn", axis=1)
y_real = sample["churn"]

X_real = pd.get_dummies(X_real, drop_first=True)
X_real = X_real.reindex(columns=X.columns, fill_value=0)
X_real_scaled = final_scaler.transform(X_real)

proba_real = final_model.predict_proba(X_real_scaled)[:, 1]
pred_real = (proba_real >= best_threshold).astype(int)

print("\n=== Evaluation on 1000-sample holdout (tuned threshold) ===")
print("Accuracy on sample:", accuracy_score(y_real, pred_real))
print("\nConfusion Matrix:\n", confusion_matrix(y_real, pred_real))
print("\nClassification Report:\n", classification_report(y_real, pred_real))
