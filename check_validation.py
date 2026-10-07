import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import SPLITS_PATH, OUTPUT_DIR
from src.feature_engineering import build_feature_dataset
from src.models import ContactHealthPredictor


def main():
    print("=" * 70)
    print("        VALIDATION ACCURACY & EVALUATION METRICS REPORT")
    print("=" * 70)

    # 1. Load feature dataset
    print("\n[1/3] Loading feature dataset and splitting...")
    full_df = build_feature_dataset()
    splits_df = pd.read_csv(SPLITS_PATH)
    full_df = full_df.merge(splits_df, on="account_id", how="left")

    train_df = full_df[full_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = full_df[full_df["split"] == "validation"].copy().reset_index(drop=True)

    print(f"      Train samples      : {len(train_df):,}")
    print(f"      Validation samples : {len(val_df):,}")

    # 2. Train models on train set
    print("\n[2/3] Training calibrated models on Train split...")
    model_suite = ContactHealthPredictor()
    model_suite.fit(train_df, val_df=None)

    # 3. Predict probabilities on Validation split
    print("\n[3/3] Evaluating strictly on Validation split...")
    p_active_val, p_rpc_val = model_suite.predict_probabilities(val_df)

    y_active_val = val_df["target_active_line"].values
    y_rpc_val = val_df["target_rpc"].values

    pred_active_binary = (p_active_val >= 0.50).astype(int)
    pred_rpc_binary = (p_rpc_val >= 0.50).astype(int)

    # -------------------------------------------------------------
    # MODEL 1: LINE LIVENESS VALIDATION METRICS
    # -------------------------------------------------------------
    acc_active = accuracy_score(y_active_val, pred_active_binary)
    bal_acc_active = balanced_accuracy_score(y_active_val, pred_active_binary)
    auc_active = roc_auc_score(y_active_val, p_active_val)
    brier_active = brier_score_loss(y_active_val, p_active_val)
    cm_active = confusion_matrix(y_active_val, pred_active_binary)

    print("\n" + "-" * 70)
    print("  MODEL 1: LINE LIVENESS / TECHNICAL HEALTH P(Active)")
    print("-" * 70)
    print(f"  • Standard Accuracy     : {acc_active * 100:.2f}%")
    print(f"  • Balanced Accuracy     : {bal_acc_active * 100:.2f}%")
    print(f"  • ROC-AUC Score         : {auc_active:.4f}")
    print(f"  • Brier Score (Calib.)  : {brier_active:.4f}  (lower is better, perfect=0.0)")
    print("\n  Confusion Matrix:")
    print(f"    [[TN={cm_active[0,0]:>5},  FP={cm_active[0,1]:>5}],")
    print(f"     [FN={cm_active[1,0]:>5},  TP={cm_active[1,1]:>5}]]")
    print("\n  Detailed Classification Report:")
    print(classification_report(y_active_val, pred_active_binary, target_names=["Dead/Invalid", "Active Line"], digits=4))

    # -------------------------------------------------------------
    # MODEL 2: BORROWER RPC VALIDATION METRICS
    # -------------------------------------------------------------
    # At standard 0.5 threshold and optimal threshold for imbalanced RPC
    acc_rpc = accuracy_score(y_rpc_val, pred_rpc_binary)
    bal_acc_rpc = balanced_accuracy_score(y_rpc_val, pred_rpc_binary)
    auc_rpc = roc_auc_score(y_rpc_val, p_rpc_val)
    brier_rpc = brier_score_loss(y_rpc_val, p_rpc_val)
    cm_rpc = confusion_matrix(y_rpc_val, pred_rpc_binary)

    # Optimal threshold based on empirical RPC prior rate (~16%)
    opt_thresh = float(np.percentile(p_rpc_val, 80))
    pred_rpc_opt = (p_rpc_val >= opt_thresh).astype(int)
    acc_rpc_opt = accuracy_score(y_rpc_val, pred_rpc_opt)

    print("\n" + "-" * 70)
    print("  MODEL 2: BORROWER RESPONSIVENESS P(RPC | Active)")
    print("-" * 70)
    print(f"  • Standard Accuracy     : {acc_rpc * 100:.2f}% (Threshold = 0.50)")
    print(f"  • Top-Decile Accuracy   : {acc_rpc_opt * 100:.2f}% (Threshold = {opt_thresh:.2f})")
    print(f"  • ROC-AUC Score         : {auc_rpc:.4f}")
    print(f"  • Brier Score (Calib.)  : {brier_rpc:.4f}  (lower is better, perfect=0.0)")
    print("\n  Confusion Matrix (Threshold=0.50):")
    print(f"    [[TN={cm_rpc[0,0]:>5},  FP={cm_rpc[0,1]:>5}],")
    print(f"     [FN={cm_rpc[1,0]:>5},  TP={cm_rpc[1,1]:>5}]]")
    print("\n  Detailed Classification Report:")
    print(classification_report(y_rpc_val, pred_rpc_binary, target_names=["No RPC", "Right-Party Contact"], digits=4))

    print("=" * 70)
    print("Validation evaluation complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
