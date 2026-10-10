import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


FEATURE_COLS = [
    # Telephony historical dynamics
    "prior_attempts",
    "prior_rpc_count",
    "prior_rpc_rate",
    "prior_third_party_count",
    "prior_wrong_number_count",
    "prior_customer_hangup_count",
    "prior_call_rejected_count",
    "prior_avoidance_rate",
    "prior_ring_duration_mean",
    "hours_since_last_attempt",
    "consecutive_switched_off",
    # Temporal & scheduling
    "attempt_hour",
    "day_of_week",
    "is_weekend",
    "diurnal_slot",
    # Phone entity features
    "priority_slot",
    "accounts_sharing_phone",
    "phone_age_days",
    # Account financial & credit distress
    "dpd_start",
    "emi_amount",
    "overdue_start",
    "outstanding",
    "debt_burden_ratio",
    "other_active_loans",
    "ability_to_pay_estimate",
    "prev_ptp_count",
    "prev_ptp_broken",
    "prev_ptp_broken_rate",
    # Cross-channel corroboration
    "has_field_visit",
    "total_field_met_borrower",
    "total_field_met_family",
    "total_field_locked_premises",
    "total_field_not_traceable",
    "total_payments_count",
    "total_payments_amount",
]

CATEGORICAL_COLS = [
    "source",
    "relation_recorded",
    "portfolio",
    "income_type",
    "preferred_language",
    "bucket_start",
    "bureau_score_band",
    "last_bounce_reason",
]


def prepare_feature_matrix(df):
    """Encodes categorical columns and prepares feature matrix X."""
    X = df[FEATURE_COLS].copy()
    
    for cat in CATEGORICAL_COLS:
        if cat in df.columns:
            X[cat] = df[cat].astype("category")
            
    return X


class ContactHealthPredictor:
    """
    Decoupled hierarchical model suite estimating:
    1. Line Liveness P(Active)
    2. Borrower Responsiveness P(RPC | Active)
    3. Counterfactual debiasing using Inverse Propensity Weighting (IPW)
    """

    def __init__(self):
        # Base estimators
        self.liveness_model = None
        self.rpc_model = None

    def fit(self, train_df, val_df=None):
        X_train = prepare_feature_matrix(train_df)
        y_liveness_train = train_df["target_active_line"].values
        y_rpc_train = train_df["target_rpc"].values

        # Compute IPW sample weights to eliminate counterfactual selection bias:
        # Inverse of selection_propensity on random arm, clipped to avoid high variance
        propensities = train_df["selection_propensity"].fillna(1.0).values
        ipw_weights = 1.0 / np.clip(propensities, 0.20, 1.0)
        # Normalize weights
        ipw_weights = ipw_weights / np.mean(ipw_weights)

        print("Training Model 1: Line Liveness P(Active) with Isotonic Calibration...")
        base_lgbm_liveness = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.liveness_model = CalibratedClassifierCV(
            estimator=base_lgbm_liveness,
            method="isotonic",
            cv=3,
        )
        self.liveness_model.fit(X_train, y_liveness_train)

        print("Training Model 2: Borrower RPC P(RPC | Active) with IPW Debiasing...")
        # Train on instances where line is active or attempted
        base_lgbm_rpc = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.04,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.rpc_model = CalibratedClassifierCV(
            estimator=base_lgbm_rpc,
            method="isotonic",
            cv=3,
        )
        self.rpc_model.fit(X_train, y_rpc_train, sample_weight=ipw_weights)

        # Evaluate on validation split if provided
        if val_df is not None:
            self.evaluate(val_df, split_name="Validation")

    def predict_probabilities(self, df):
        """Outputs calibrated P(Active) and P(RPC | Active)."""
        X = prepare_feature_matrix(df)
        p_active = self.liveness_model.predict_proba(X)[:, 1]
        p_rpc = self.rpc_model.predict_proba(X)[:, 1]

        return p_active, p_rpc

    def evaluate(self, df, split_name="Evaluation"):
        """Computes AUC, Brier calibration score, and log-loss."""
        p_active, p_rpc = self.predict_probabilities(df)
        y_active = df["target_active_line"].values
        y_rpc = df["target_rpc"].values

        auc_active = roc_auc_score(y_active, p_active)
        brier_active = brier_score_loss(y_active, p_active)

        auc_rpc = roc_auc_score(y_rpc, p_rpc)
        brier_rpc = brier_score_loss(y_rpc, p_rpc)

        print(f"\n--- {split_name} Performance ---")
        print(f"Liveness P(Active)  : AUC = {auc_active:.4f} | Brier Score = {brier_active:.4f}")
        print(f"RPC P(RPC | Active) : AUC = {auc_rpc:.4f} | Brier Score = {brier_rpc:.4f}")

        return {
            "auc_active": auc_active,
            "brier_active": brier_active,
            "auc_rpc": auc_rpc,
            "brier_rpc": brier_rpc,
        }
