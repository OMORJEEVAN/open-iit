import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np

                          
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
from src.field_policy_engine import FieldPolicyEngine
from src.skip_trace_optimizer import SkipTraceOptimizer


def main():
    print("======================================================================")
    print("  CREDITNIRVANA PS2: RIGHT-PARTY CONTACT & SKIP-TRACE OPTIMIZER      ")
    print("======================================================================\n")

                                 
    full_df = build_feature_dataset()

                                                  
    splits_df = pd.read_csv(SPLITS_PATH)
    full_df = full_df.merge(splits_df, on="account_id", how="left")

    train_df = full_df[full_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = full_df[full_df["split"] == "validation"].copy().reset_index(drop=True)
    test_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)

    print(f"\nDataset Splits:")
    print(f"Train attempts: {len(train_df):,} | Val attempts: {len(val_df):,} | Test attempts: {len(test_df):,}")

                                              
    print("\n--- Training Hierarchical ML Models ---")
    model_suite = ContactHealthPredictor()
    model_suite.fit(train_df, val_df)

                                           
    print("\n--- Evaluating on Out-Of-Sample Test Split ---")
    test_metrics = model_suite.evaluate(test_df, split_name="Test Set")

                                                     
    p_active_test, p_rpc_test = model_suite.predict_probabilities(test_df)
    p_active_full, p_rpc_full = model_suite.predict_probabilities(full_df)

                                              
    phones_df = pd.read_csv(PHONES_PATH)
    acc_phones_map = phones_df.groupby("account_id")["phone_id"].unique().to_dict()

                                                            
    print("\n--- Applying Telephony Policy Engine (State-to-Action Mapping) ---")
    engine = PolicyEngine()
    test_policy = engine.prescribe_actions(test_df, p_active_test, p_rpc_test, acc_phones_map)
    full_policy = engine.prescribe_actions(full_df, p_active_full, p_rpc_full, acc_phones_map)

    print("\nDistribution of Predicted Contact States (Test Set):")
    state_counts = test_policy["predicted_state"].value_counts()
    for state, count in state_counts.items():
        print(f"  • {state:<40}: {count:>5} ({count/len(test_policy)*100:>5.1f}%)")

    print("\nDistribution of Prescribed Operational Actions (Test Set):")
    action_counts = test_policy["prescribed_action"].value_counts()
    for action, count in action_counts.items():
        print(f"  • {action:<55}: {count:>5} ({count/len(test_policy)*100:>5.1f}%)")

                                                                                           
    print("\n--- Applying Address & Field Visit Policy Engine (PS2 - Field Channel) ---")
    field_engine = FieldPolicyEngine()
    address_policy = field_engine.evaluate_addresses()

    print("\nDistribution of Address States & Prescribed Field Actions:")
    addr_state_counts = address_policy["predicted_state"].value_counts()
    for state, count in addr_state_counts.items():
        print(f"  • {state:<45}: {count:>5} ({count/len(address_policy)*100:>5.1f}%)")

    addr_action_counts = address_policy["prescribed_action"].value_counts()
    print("\nPrescribed Field Operations Actions:")
    for action, count in addr_action_counts.items():
        print(f"  • {action:<55}: {count:>5} ({count/len(address_policy)*100:>5.1f}%)")

                                                                                            
    print("\n--- Ground Truth Validation Against verified_contact_points.csv (Full Audit) ---")
    vcp_df = pd.read_csv(VERIFIED_CONTACTS_PATH)
    latest_full_preds = (
        full_policy.sort_values("attempt_ts")
        .groupby(["phone_id", "account_id"])
        .last()
        .reset_index()
    )
    merged_vcp = vcp_df.merge(latest_full_preds, on=["phone_id", "account_id"], how="inner")
    crosstab = pd.crosstab(
        merged_vcp["verified_status"],
        merged_vcp["predicted_state"],
        margins=True
    )
    print("\nAudit Matrix across all audited contact points (250 contacts):")
    print(crosstab)
    merged_vcp.to_csv(os.path.join(OUTPUT_DIR, "verified_audit_comparison.csv"), index=False)

                                                                
    print("\n--- Optimizing Skip-Trace Queue via Value-of-Information (VoI) ---")
    accounts_df = pd.read_csv(ACCOUNTS_PATH)
    optimizer = SkipTraceOptimizer()
    optimizer.fit()

                                                                                  
    test_account_ids = test_df["account_id"].unique()
    test_trace_queue = optimizer.optimize_queue(
        accounts_df,
        policy_df=test_policy,
        account_ids_filter=test_account_ids,
        address_policy_df=address_policy,
    )
    print(f"\nPrioritized Skip-Trace Queue (Test Set): {len(test_trace_queue)} accounts eligible for tracing.")
    if len(test_trace_queue) > 0:
        print("Top Accounts in Test Queue Ranked by Expected Net Recovery Value:")
        print(
            test_trace_queue[
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

                                                       
    full_trace_queue = optimizer.optimize_queue(
        accounts_df,
        policy_df=full_policy,
        address_policy_df=address_policy,
    )
    print(f"\nPortfolio-Wide Skip-Trace Queue (All Accounts): {len(full_trace_queue)} accounts eligible for tracing.")

                  
    test_policy.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_policy.csv"), index=False)
    test_trace_queue.to_csv(os.path.join(OUTPUT_DIR, "test_skip_trace_priority_queue.csv"), index=False)
    full_trace_queue.to_csv(os.path.join(OUTPUT_DIR, "skip_trace_priority_queue.csv"), index=False)
    address_policy.to_csv(os.path.join(OUTPUT_DIR, "address_field_predictions.csv"), index=False)

    print(f"\nAll artifacts successfully saved to '{OUTPUT_DIR}/':")
    print("  1. test_predictions_with_policy.csv")
    print("  2. address_field_predictions.csv")
    print("  3. test_skip_trace_priority_queue.csv")
    print("  4. skip_trace_priority_queue.csv")
    print("  5. verified_audit_comparison.csv")
    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()
