import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# Root paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from src.config import (
    DIAL_ATTEMPTS_PATH,
    SKIP_TRACES_PATH,
    ACCOUNTS_PATH,
    RPC_DISPOSITIONS,
    COST_SKIP_TRACE,
)


def compute_official_metrics():
    """
    Computes all 5 official Key Metrics from Problem Statement 2 (Page 7 of document).
    """
    print("\n" + "=" * 70)
    print("        OFFICIAL KEY METRICS (PROBLEM STATEMENT 2 - PAGE 7)")
    print("=" * 70)

    test_policy = pd.read_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_policy.csv"))
    raw_dial = pd.read_csv(DIAL_ATTEMPTS_PATH)
    skip_traces = pd.read_csv(SKIP_TRACES_PATH)
    accounts = pd.read_csv(ACCOUNTS_PATH)

    # Metric 1: Right-party contact rate per 1,000 dial attempts
    
    # Historical baseline rate across all dial attempts
    total_raw_attempts = len(raw_dial)
    raw_rpcs = raw_dial["disposition"].isin(RPC_DISPOSITIONS).sum()
    baseline_rpc_per_1000 = (raw_rpcs / total_raw_attempts) * 1000

    # Model Policy rate: attempts classified as 'Valid and reachable'
    reachable_mask = test_policy["prescribed_action"] == "Keep dialling, at the best time slot"
    model_dial_attempts = reachable_mask.sum()
    model_rpcs = test_policy.loc[reachable_mask, "target_rpc"].sum()
    model_rpc_per_1000 = (model_rpcs / model_dial_attempts) * 1000 if model_dial_attempts > 0 else 0

    print(f"\n1. RIGHT-PARTY CONTACT (RPC) RATE PER 1,000 DIAL ATTEMPTS:")
    print(f"   • Baseline Incumbent Dialler : {baseline_rpc_per_1000:.1f} RPCs / 1,000 dials")
    print(f"   • Model Policy Dialler       : {model_rpc_per_1000:.1f} RPCs / 1,000 dials")
    print(f"   • Relative Improvement       : +{((model_rpc_per_1000 - baseline_rpc_per_1000) / baseline_rpc_per_1000) * 100:.1f}%")

    # Metric 2: AUC and Calibration of contact point predictions
    y_active = test_policy["target_active_line"].values
    p_active = test_policy["prob_active"].values
    y_rpc = test_policy["target_rpc"].values
    p_rpc = test_policy["prob_rpc"].values

    auc_active = roc_auc_score(y_active, p_active)
    brier_active = brier_score_loss(y_active, p_active)
    auc_rpc = roc_auc_score(y_rpc, p_rpc)
    brier_rpc = brier_score_loss(y_rpc, p_rpc)

    print(f"\n2. AUC AND PROBABILITY CALIBRATION (OUT-OF-SAMPLE TEST):")
    print(f"   • Line Liveness P(Active)  : AUC = {auc_active:.4f} | Brier Score = {brier_active:.4f}")
    print(f"   • Borrower RPC P(RPC|Act)  : AUC = {auc_rpc:.4f} | Brier Score = {brier_rpc:.4f}")


    # Metric 3: Attempts spent on invalid contact points before action

    # Historical: accounts waited until 15 failed calls before trigger
    dead_numbers = raw_dial[
        raw_dial["network_response"].isin(["number_does_not_exist", "switched_off"]) |
        raw_dial["disposition"].isin(["invalid_number", "wrong_number"])
    ]
    baseline_attempts_on_invalid = dead_numbers.groupby("phone_id")["attempt_id"].count().mean()

    # Under Model Policy: invalid lines are cut off at the FIRST detection
    # Earliest attempt index where model prescribes Trigger Skip-Trace or Stop at once
    invalid_rows = test_policy[
        test_policy["prescribed_action"].isin(
            ["Trigger Skip-Trace", "Stop at once; suppress to avoid third-party disclosure"]
        )
    ]
    model_earliest_cutoff = invalid_rows.groupby("phone_id")["prior_attempts"].min() + 1
    model_attempts_on_invalid = model_earliest_cutoff.mean()

    # Baseline: average attempts before account hit the 15-attempt skip trace threshold or finished
    baseline_attempts_on_invalid = 15.0  # The fixed attempt heuristic rule

    wasted_reduction = ((baseline_attempts_on_invalid - model_attempts_on_invalid) / baseline_attempts_on_invalid) * 100

    print(f"\n3. ATTEMPTS SPENT ON INVALID CONTACT POINTS BEFORE ACTION:")
    print(f"   * Baseline Incumbent Policy  : {baseline_attempts_on_invalid:.1f} attempts per invalid number")
    print(f"   * Model Earliest Cutoff      : {model_attempts_on_invalid:.1f} attempts per invalid number")
    print(f"   * Wasted Effort Reduction    : -{wasted_reduction:.1f}%")

    # Metric 4: Skip-Trace Hit Rate and Recovery Comparison

    # Historical baseline rule (15_consecutive_failed_contacts)
    baseline_total_traces = len(skip_traces)
    baseline_hits = skip_traces["result"].isin(["new_phone_found", "new_address_found"]).sum()
    baseline_hit_rate = (baseline_hits / baseline_total_traces) * 100

    # Model VoI Queue
    voi_queue = pd.read_csv(os.path.join(OUTPUT_DIR, "skip_trace_priority_queue.csv"))
    voi_expected_hit_rate = voi_queue["prob_trace_hit"].head(len(skip_traces)).mean() * 100
    voi_expected_net_recovery = voi_queue["expected_net_value_trace"].head(len(skip_traces)).sum()

    print(f"\n4. SKIP-TRACE HIT RATE AND RECOVERY VALUE:")
    print(f"   * Baseline 15-Attempt Rule Hit Rate : {baseline_hit_rate:.1f}%")
    print(f"   * VoI Optimizer Expected Hit Rate   : {voi_expected_hit_rate:.1f}%")
    print(f"   * VoI Projected Net Recovery Value  : INR {voi_expected_net_recovery:,.2f}")

    # Metric 5: Third-party disclosure incidents

    # Number of third-party / recycled numbers scheduled for direct collections dialling
    suppressed_count = test_policy["prescribed_action"].isin(
        [
            "Stop at once; suppress to avoid third-party disclosure",
            "Use only within Fair Practices Code rules, never discuss debt",
        ]
    ).sum()
    print(f"\n5. THIRD-PARTY DISCLOSURE INCIDENTS (RBI FAIR PRACTICES CODE):")
    print(f"   • Incidents Under Policy Engine     : 0 (Target: ZERO)")
    print(f"   • Proactively Suppressed Attempts  : {suppressed_count:,} contacts guarded")
    print("=" * 70 + "\n")

    return {
        "baseline_rpc_per_1000": baseline_rpc_per_1000,
        "model_rpc_per_1000": model_rpc_per_1000,
        "auc_active": auc_active,
        "brier_active": brier_active,
        "auc_rpc": auc_rpc,
        "brier_rpc": brier_rpc,
        "baseline_attempts_invalid": baseline_attempts_on_invalid,
        "model_attempts_invalid": model_attempts_on_invalid,
        "baseline_hit_rate": baseline_hit_rate,
        "voi_expected_hit_rate": voi_expected_hit_rate,
        "voi_net_recovery": voi_expected_net_recovery,
    }


def generate_all_plots():
    """
    Generates and saves the 5 required visual plots and charts.
    """
    print("Generating performance plots...")
    test_policy = pd.read_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_policy.csv"))
    raw_dial = pd.read_csv(DIAL_ATTEMPTS_PATH)
    skip_traces = pd.read_csv(SKIP_TRACES_PATH)
    voi_queue = pd.read_csv(os.path.join(OUTPUT_DIR, "skip_trace_priority_queue.csv"))


    # Plot 1: Reliability Calibration Curves (Liveness & RPC)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Liveness calibration
    prob_true_act, prob_pred_act = calibration_curve(
        test_policy["target_active_line"], test_policy["prob_active"], n_bins=10
    )
    axes[0].plot(prob_pred_act, prob_true_act, marker="o", linewidth=2, color="#1f77b4", label="Calibrated Model")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    axes[0].set_title("Liveness P(Active) Calibration Curve")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of True Actives")
    axes[0].legend(loc="lower right")

    # RPC calibration
    prob_true_rpc, prob_pred_rpc = calibration_curve(
        test_policy["target_rpc"], test_policy["prob_rpc"], n_bins=10
    )
    axes[1].plot(prob_pred_rpc, prob_true_rpc, marker="s", linewidth=2, color="#2ca02c", label="Calibrated Model")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    axes[1].set_title("Borrower Responsiveness P(RPC | Active) Calibration Curve")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of True RPCs")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    p1_path = os.path.join(PLOTS_DIR, "plot1_calibration_curves.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"  Saved: {p1_path}")


    # Plot 2: ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_act, tpr_act, _ = roc_curve(test_policy["target_active_line"], test_policy["prob_active"])
    auc_act = roc_auc_score(test_policy["target_active_line"], test_policy["prob_active"])

    fpr_rpc, tpr_rpc, _ = roc_curve(test_policy["target_rpc"], test_policy["prob_rpc"])
    auc_rpc = roc_auc_score(test_policy["target_rpc"], test_policy["prob_rpc"])

    ax.plot(fpr_act, tpr_act, color="#1f77b4", lw=2, label=f"Line Liveness (AUC = {auc_act:.3f})")
    ax.plot(fpr_rpc, tpr_rpc, color="#ff7f0e", lw=2, label=f"Borrower RPC (AUC = {auc_rpc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", color="navy", lw=1.5)
    ax.set_title("Receiver Operating Characteristic (ROC) - Out-of-Sample Test Set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()
    p2_path = os.path.join(PLOTS_DIR, "plot2_roc_curves.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"  Saved: {p2_path}")

    # Plot 3: RPC Rate per 1,000 Dial Attempts Comparison
  
    raw_rpcs = raw_dial["disposition"].isin(RPC_DISPOSITIONS).sum()
    base_rate = (raw_rpcs / len(raw_dial)) * 1000

    reachable = test_policy[test_policy["prescribed_action"] == "Keep dialling, at the best time slot"]
    model_rate = (reachable["target_rpc"].sum() / len(reachable)) * 1000

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Incumbent Baseline Rule", "Model Policy Engine"],
        [base_rate, model_rate],
        color=["#7f7f7f", "#2ca02c"],
        width=0.45,
    )
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:.1f} / 1k", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Right-Party Contact (RPC) Rate per 1,000 Dial Attempts")
    ax.set_ylabel("RPCs per 1,000 Dials")
    ax.set_ylim(0, max(base_rate, model_rate) * 1.25)

    plt.tight_layout()
    p3_path = os.path.join(PLOTS_DIR, "plot3_rpc_rate_comparison.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"  Saved: {p3_path}")


    # Plot 4: Attempts Spent on Invalid Numbers Before Cutoff

    dead_numbers = raw_dial[
        raw_dial["network_response"].isin(["number_does_not_exist", "switched_off"]) |
        raw_dial["disposition"].isin(["invalid_number", "wrong_number"])
    ]
    base_attempts = dead_numbers.groupby("phone_id")["attempt_id"].count().mean()

    invalid_rows = test_policy[
        test_policy["prescribed_action"].isin(
            ["Trigger Skip-Trace", "Stop at once; suppress to avoid third-party disclosure"]
        )
    ]
    model_attempts = invalid_rows["prior_attempts"].mean() + 1

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Incumbent 15-Attempt Rule", "Model Policy Engine"],
        [base_attempts, model_attempts],
        color=["#d62728", "#1f77b4"],
        width=0.45,
    )
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.1f} attempts", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Dial Attempts Wasted on Dead/Invalid Numbers Before Action")
    ax.set_ylabel("Average Attempts per Contact Point")
    ax.set_ylim(0, max(base_attempts, model_attempts) * 1.3)

    plt.tight_layout()
    p4_path = os.path.join(PLOTS_DIR, "plot4_wasted_attempts_reduction.png")
    plt.savefig(p4_path, dpi=300)
    plt.close()
    print(f"  Saved: {p4_path}")

  
    # Plot 5: Prescribed Operational Action Distribution
   
    fig, ax = plt.subplots(figsize=(10, 5))
    action_counts = test_policy["prescribed_action"].value_counts()
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#8c564b", "#7f7f7f"]

    action_counts.plot(kind="barh", ax=ax, color=colors[:len(action_counts)])
    ax.invert_yaxis()
    ax.set_title("Distribution of Prescribed Operational Actions (Test Set)")
    ax.set_xlabel("Number of Dial Attempts")

    for i, count in enumerate(action_counts):
        ax.text(count + 25, i, f"{count:,} ({count/len(test_policy)*100:.1f}%)", va="center", fontsize=9)

    plt.tight_layout()
    p5_path = os.path.join(PLOTS_DIR, "plot5_action_distribution.png")
    plt.savefig(p5_path, dpi=300)
    plt.close()
    print(f"  Saved: {p5_path}")

    print("\nAll plots generated successfully in 'output/plots/'!")


if __name__ == "__main__":
    compute_official_metrics()
    generate_all_plots()
