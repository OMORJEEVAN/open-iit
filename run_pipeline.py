import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SPLITS_PATH,
    VERIFIED_CONTACTS_PATH,
    ACCOUNTS_PATH,
    PHONES_PATH,
    OUTPUT_DIR,
)
from src.feature_engineering import build_feature_dataset
from src.models import ContactHealthPredictor
from src.policy_engine import PolicyEngine
from src.skip_trace_optimizer import SkipTraceOptimizer


def main():
    print("======================================================================")
    print("  CREDITNIRVANA PS2: RIGHT-PARTY CONTACT & SKIP-TRACE OPTIMIZER      ")
    print("======================================================================\n")

    # Step 1: Feature Engineering
    full_df = build_feature_dataset()

    # Step 2: Split according to shared/splits.csv
    splits_df = pd.read_csv(SPLITS_PATH)
    full_df = full_df.merge(splits_df, on="account_id", how="left")

    train_df = full_df[full_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = full_df[full_df["split"] == "validation"].copy().reset_index(drop=True)
    test_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)

    print(f"\nDataset Splits:")
    print(f"Train attempts: {len(train_df):,} | Val attempts: {len(val_df):,} | Test attempts: {len(test_df):,}")

    # Step 3: Train Contact Health Model Suite
    print("\n--- Training Hierarchical ML Models ---")
    model_suite = ContactHealthPredictor()
    model_suite.fit(train_df, val_df)

    # Step 4: Evaluate on Unseen Test Split
    print("\n--- Evaluating on Out-Of-Sample Test Split ---")
    test_metrics = model_suite.evaluate(test_df, split_name="Test Set")

    # Predict probabilities for test set
    p_active_test, p_rpc_test = model_suite.predict_probabilities(test_df)

    # Prepare account-phones map for cascading
    phones_df = pd.read_csv(PHONES_PATH)
    acc_phones_map = phones_df.groupby("account_id")["phone_id"].unique().to_dict()

    # Step 5: Run Policy Engine (Next Best Action)
    print("\n--- Applying Policy Engine (State-to-Action Mapping) ---")
    engine = PolicyEngine()
    test_policy = engine.prescribe_actions(test_df, p_active_test, p_rpc_test, acc_phones_map)

    print("\nDistribution of Predicted Contact States (Test Set):")
    state_counts = test_policy["predicted_state"].value_counts()
    for state, count in state_counts.items():
        print(f"  • {state:<40}: {count:>5} ({count/len(test_policy)*100:>5.1f}%)")

    print("\nDistribution of Prescribed Operational Actions (Test Set):")
    action_counts = test_policy["prescribed_action"].value_counts()
    for action, count in action_counts.items():
        print(f"  • {action:<55}: {count:>5} ({count/len(test_policy)*100:>5.1f}%)")

    # Step 6: Validate against verified_contact_points.csv Ground Truth
    print("\n--- Ground Truth Validation Against verified_contact_points.csv ---")
    vcp_df = pd.read_csv(VERIFIED_CONTACTS_PATH)
    
    # Get the latest pre-contact prediction per phone
    latest_phone_preds = (
        test_policy.sort_values("attempt_ts")
        .groupby(["phone_id", "account_id"])
        .last()
        .reset_index()
    )

    merged_vcp = vcp_df.merge(latest_phone_preds, on=["phone_id", "account_id"], how="inner")
    if len(merged_vcp) > 0:
        crosstab = pd.crosstab(
            merged_vcp["verified_status"],
            merged_vcp["predicted_state"],
            margins=True
        )
        print("\nAudit Matrix (Verified Status vs. Predicted State):")
        print(crosstab)
        merged_vcp.to_csv(os.path.join(OUTPUT_DIR, "verified_audit_comparison.csv"), index=False)
    else:
        print("Note: Verified contact points span accounts across splits. Evaluating full dataset coverage...")
        # Full predictions for complete audit check
        p_active_full, p_rpc_full = model_suite.predict_probabilities(full_df)
        full_policy = engine.prescribe_actions(full_df, p_active_full, p_rpc_full, acc_phones_map)
        latest_full_preds = (
            full_policy.sort_values("attempt_ts")
            .groupby(["phone_id", "account_id"])
            .last()
            .reset_index()
        )
        merged_vcp_full = vcp_df.merge(latest_full_preds, on=["phone_id", "account_id"], how="inner")
        crosstab = pd.crosstab(
            merged_vcp_full["verified_status"],
            merged_vcp_full["predicted_state"],
            margins=True
        )
        print("\nAudit Matrix across all audited contact points:")
        print(crosstab)
        merged_vcp_full.to_csv(os.path.join(OUTPUT_DIR, "verified_audit_comparison.csv"), index=False)

    # Step 7: Value-of-Information (VoI) Skip-Trace Optimization
    print("\n--- Optimizing Skip-Trace Queue via Value-of-Information (VoI) ---")
    accounts_df = pd.read_csv(ACCOUNTS_PATH)
    optimizer = SkipTraceOptimizer()
    optimizer.fit()

    trace_queue = optimizer.optimize_queue(accounts_df, test_policy)
    print(f"\nPrioritized Skip-Trace Queue Generated: {len(trace_queue)} accounts eligible for tracing.")
    print("Top 5 Accounts Ranked by Expected Net Recovery Value:")
    print(
        trace_queue[
            [
                "account_id",
                "portfolio",
                "outstanding",
                "ability_to_pay_estimate",
                "prob_trace_hit",
                "expected_net_value_trace",
            ]
        ].head(5).to_string(index=False)
    )

    # Save outputs
    test_policy.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_policy.csv"), index=False)
    trace_queue.to_csv(os.path.join(OUTPUT_DIR, "skip_trace_priority_queue.csv"), index=False)

    print(f"\nAll artifacts successfully saved to '{OUTPUT_DIR}/':")
    print("  1. test_predictions_with_policy.csv")
    print("  2. skip_trace_priority_queue.csv")
    print("  3. verified_audit_comparison.csv")
    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()
