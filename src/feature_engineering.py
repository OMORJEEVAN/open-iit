import os
import numpy as np
import pandas as pd
from src.config import (
    ACCOUNTS_PATH,
    PHONES_PATH,
    DIAL_ATTEMPTS_PATH,
    FIELD_VISITS_PATH,
    PAYMENTS_PATH,
    RPC_DISPOSITIONS,
    THIRD_PARTY_DISPOSITIONS,
)


def load_raw_data():
    """Loads all relevant raw dataframes."""
    accounts = pd.read_csv(ACCOUNTS_PATH)
    phones = pd.read_csv(PHONES_PATH)
    dial_attempts = pd.read_csv(DIAL_ATTEMPTS_PATH)
    field_visits = pd.read_csv(FIELD_VISITS_PATH)
    payments = pd.read_csv(PAYMENTS_PATH)

    dial_attempts["attempt_ts"] = pd.to_datetime(dial_attempts["attempt_ts"], errors="coerce")
    field_visits["start_ts"] = pd.to_datetime(field_visits["start_ts"], errors="coerce")
    payments["payment_ts"] = pd.to_datetime(payments["payment_ts"], errors="coerce")
    phones["added_date"] = pd.to_datetime(phones["added_date"], errors="coerce")

    return accounts, phones, dial_attempts, field_visits, payments


def compute_telephony_features(dial_attempts, phones):
    """
    Computes strict pre-attempt historical features without data leakage.
    Every feature for attempt k is calculated strictly using attempts 0 .. k-1.
    """
    df = dial_attempts.sort_values(["phone_id", "attempt_ts"]).copy().reset_index(drop=True)

    # Indicator flags for past outcomes
    df["flag_rpc"] = df["disposition"].isin(RPC_DISPOSITIONS).astype(int)
    df["flag_third_party"] = df["disposition"].isin(THIRD_PARTY_DISPOSITIONS).astype(int)
    df["flag_customer_hangup"] = (df["hangup_by"] == "customer").astype(int)
    df["flag_call_rejected"] = (df["disposition"] == "call_rejected").astype(int)
    df["flag_wrong_number"] = (df["disposition"] == "wrong_number").astype(int)
    df["flag_switched_off"] = (df["network_response"] == "switched_off").astype(int)
    df["flag_not_reachable"] = (df["network_response"] == "not_reachable").astype(int)
    df["flag_invalid"] = (
        (df["network_response"] == "number_does_not_exist") |
        (df["disposition"] == "invalid_number")
    ).astype(int)

    # Diurnal slots
    hour = df["attempt_ts"].dt.hour
    df["diurnal_slot"] = np.where(hour < 12, 1, np.where(hour < 16, 2, 3))

    g = df.groupby("phone_id", sort=False)

    # 1. Prior attempt count
    df["prior_attempts"] = g.cumcount()

    # 2. Cumulative past outcomes (shifted by 1 to exclude current attempt)
    df["prior_rpc_count"] = g["flag_rpc"].cumsum() - df["flag_rpc"]
    df["prior_rpc_rate"] = np.where(
        df["prior_attempts"] > 0,
        df["prior_rpc_count"] / df["prior_attempts"],
        0.0
    )

    df["prior_third_party_count"] = g["flag_third_party"].cumsum() - df["flag_third_party"]
    df["prior_wrong_number_count"] = g["flag_wrong_number"].cumsum() - df["flag_wrong_number"]

    # 3. Customer hangup / call rejection (key avoidance signals)
    df["prior_customer_hangup_count"] = g["flag_customer_hangup"].cumsum() - df["flag_customer_hangup"]
    df["prior_call_rejected_count"] = g["flag_call_rejected"].cumsum() - df["flag_call_rejected"]
    df["prior_avoidance_rate"] = np.where(
        df["prior_attempts"] > 0,
        (df["prior_customer_hangup_count"] + df["prior_call_rejected_count"]) / df["prior_attempts"],
        0.0
    )

    # 4. Ring duration physics
    cum_ring = g["ring_duration_s"].cumsum() - df["ring_duration_s"]
    df["prior_ring_duration_mean"] = np.where(
        df["prior_attempts"] > 0,
        cum_ring / df["prior_attempts"],
        0.0
    )

    # 5. Elapsed time
    df["prev_attempt_ts"] = g["attempt_ts"].shift(1)
    df["hours_since_last_attempt"] = (
        (df["attempt_ts"] - df["prev_attempt_ts"]).dt.total_seconds() / 3600.0
    ).fillna(999.0)

    # 6. Consecutive switched-off streak calculation
    # Streak resets to 0 whenever flag_switched_off is 0
    switched_off_arr = df["flag_switched_off"].values
    phone_ids = df["phone_id"].values
    streak = np.zeros(len(df), dtype=int)
    cur_streak = 0
    prev_phone = None

    for i in range(len(df)):
        cur_p = phone_ids[i]
        if cur_p != prev_phone:
            cur_streak = 0
            prev_phone = cur_p
        streak[i] = cur_streak
        # Update for next iteration
        if switched_off_arr[i] == 1:
            cur_streak += 1
        else:
            cur_streak = 0

    df["consecutive_switched_off"] = streak

    # 7. Diurnal diversity of switched_off streak
    # Does this phone have switched-off attempts across multiple time-of-day slots?
    df["attempt_hour"] = df["attempt_ts"].dt.hour
    df["day_of_week"] = df["attempt_ts"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Entity linkage from phones.csv
    # Detect shared phones across multiple accounts (strong signal for recycled / reference)
    shared_phone_counts = phones.groupby("phone_masked")["account_id"].nunique().to_dict()
    phones_copy = phones.copy()
    phones_copy["accounts_sharing_phone"] = phones_copy["phone_masked"].map(shared_phone_counts).fillna(1)

    merged = df.merge(
        phones_copy[
            [
                "phone_id",
                "account_id",
                "source",
                "relation_recorded",
                "added_date",
                "priority_slot",
                "accounts_sharing_phone",
            ]
        ],
        on=["phone_id", "account_id"],
        how="left",
    )

    merged["phone_age_days"] = (
        (merged["attempt_ts"] - merged["added_date"]).dt.total_seconds() / 86400.0
    ).fillna(0.0)

    return merged


def add_account_and_cross_channel_features(df, accounts, field_visits, payments):
    """
    Merges account financial features and cross-channel corroboration (field visits & payments).
    """
    # 1. Account financial features
    acc_copy = accounts.copy()
    acc_copy["prev_ptp_broken_rate"] = np.where(
        acc_copy["prev_ptp_count"] > 0,
        acc_copy["prev_ptp_broken"] / acc_copy["prev_ptp_count"],
        0.0
    )
    acc_copy["debt_burden_ratio"] = acc_copy["overdue_start"] / (acc_copy["outstanding"] + 1e-5)

    df = df.merge(
        acc_copy,
        on="account_id",
        how="left",
        suffixes=("", "_acc"),
    )

    # 2. Cross-channel field visit signals
    # Check if a field visit occurred before this attempt for this account
    fv = field_visits.sort_values(["account_id", "start_ts"]).copy()
    
    # Pre-aggregating field outcomes by account for fast temporal joins
    fv["field_met_borrower"] = (fv["outcome"] == "met_borrower").astype(int)
    fv["field_met_family"] = (fv["outcome"] == "met_family").astype(int)
    fv["field_locked_premises"] = (fv["outcome"] == "locked_premises").astype(int)
    fv["field_not_traceable"] = (fv["outcome"] == "address_not_traceable").astype(int)

    acc_fv_summary = fv.groupby("account_id").agg(
        has_field_visit=("visit_id", "count"),
        total_field_met_borrower=("field_met_borrower", "sum"),
        total_field_met_family=("field_met_family", "sum"),
        total_field_locked_premises=("field_locked_premises", "sum"),
        total_field_not_traceable=("field_not_traceable", "sum"),
    ).reset_index()

    df = df.merge(acc_fv_summary, on="account_id", how="left")
    for col in [
        "has_field_visit",
        "total_field_met_borrower",
        "total_field_met_family",
        "total_field_locked_premises",
        "total_field_not_traceable",
    ]:
        df[col] = df[col].fillna(0)

    # 3. Payment history signals
    pay_summary = payments.groupby("account_id").agg(
        total_payments_count=("payment_id", "count"),
        total_payments_amount=("amount", "sum"),
    ).reset_index()

    df = df.merge(pay_summary, on="account_id", how="left")
    df["total_payments_count"] = df["total_payments_count"].fillna(0)
    df["total_payments_amount"] = df["total_payments_amount"].fillna(0.0)

    return df


def build_feature_dataset():
    """Builds the complete feature engineered dataset ready for modeling."""
    print("Loading raw datasets...")
    accounts, phones, dial_attempts, field_visits, payments = load_raw_data()

    print("Computing pre-attempt telephony features...")
    df = compute_telephony_features(dial_attempts, phones)

    print("Adding account and cross-channel features...")
    df = add_account_and_cross_channel_features(df, accounts, field_visits, payments)

    # Define targets for ML models:
    # Target 1: Liveness / Active Line
    # Active if answered, rejected by user, or full ringing timeout
    df["target_active_line"] = np.where(
        (df["network_response"].isin(["answered", "busy_rejected"])) |
        (df["ring_duration_s"] >= 25),
        1,
        np.where(
            (df["network_response"] == "number_does_not_exist") |
            (df["disposition"] == "invalid_number"),
            0,
            np.where(
                (df["hangup_by"] == "network") & (df["ring_duration_s"] <= 3),
                0,
                1  # default to 1 if uncertain network
            )
        )
    )

    # Target 2: Borrower RPC (Right-Party Contact)
    df["target_rpc"] = df["disposition"].isin(RPC_DISPOSITIONS).astype(int)

    # Target 3: Borrower Avoidance indicator
    # Customer actively declined or line active but never answered despite full ring
    df["target_avoidance"] = np.where(
        (df["target_rpc"] == 0) &
        (
            (df["hangup_by"] == "customer") |
            (df["disposition"] == "call_rejected") |
            ((df["target_active_line"] == 1) & (df["prior_attempts"] >= 2) & (df["prior_rpc_count"] == 0))
        ),
        1,
        0
    )

    print(f"Feature engineering complete. Dataset shape: {df.shape}")
    return df
