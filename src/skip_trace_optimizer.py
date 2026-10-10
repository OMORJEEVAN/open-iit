import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.config import SKIP_TRACES_PATH, ACCOUNTS_PATH, COST_SKIP_TRACE


class SkipTraceOptimizer:
    """
    Ranks accounts for skip-tracing using an Expected Value-of-Information (VoI)
    economic framework instead of fixed attempt heuristics.
    """

    def __init__(self):
        self.hit_model = None

    def fit(self):
        skip_traces = pd.read_csv(SKIP_TRACES_PATH)
        accounts = pd.read_csv(ACCOUNTS_PATH)

                                                       
        skip_traces["target_hit"] = skip_traces["result"].isin(
            ["new_phone_found", "new_address_found"]
        ).astype(int)

        merged = skip_traces.merge(accounts, on="account_id", how="left")

        features = [
            "dpd_start",
            "emi_amount",
            "overdue_start",
            "outstanding",
            "ability_to_pay_estimate",
            "prev_ptp_count",
            "prev_ptp_broken",
            "other_active_loans",
        ]
        
        X = merged[features].fillna(0)
        y = merged["target_hit"].values

        base_lgbm = LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            random_state=42,
            verbose=-1,
        )
        self.hit_model = CalibratedClassifierCV(
            estimator=base_lgbm,
            method="isotonic",
            cv=3,
        )
        self.hit_model.fit(X, y)
        print("SkipTraceOptimizer: Hit probability model fitted successfully.")

    def optimize_queue(self, accounts_df, policy_df=None, account_ids_filter=None, address_policy_df=None):
        """
        Calculates Expected Net Value of Trace (ENVT) and ranks accounts.
        """
        if account_ids_filter is not None:
            accounts_df = accounts_df[accounts_df["account_id"].isin(account_ids_filter)].copy()

        features = [
            "dpd_start",
            "emi_amount",
            "overdue_start",
            "outstanding",
            "ability_to_pay_estimate",
            "prev_ptp_count",
            "prev_ptp_broken",
            "other_active_loans",
        ]
        X = accounts_df[features].fillna(0)
        p_hit = self.hit_model.predict_proba(X)[:, 1]

        acc_queue = accounts_df.copy()
        acc_queue["prob_trace_hit"] = p_hit

                                                                    
        p_rpc_given_found = 0.45

                                                                                      
        expected_recovery = (
            acc_queue["outstanding"].fillna(0) *
            acc_queue["ability_to_pay_estimate"].fillna(0.5) *
            0.60                                         
        )

        acc_queue["expected_recovery_gross"] = expected_recovery
        acc_queue["expected_net_value_trace"] = (
            acc_queue["prob_trace_hit"] * p_rpc_given_found * expected_recovery
            - COST_SKIP_TRACE
        )

                                               
                                                                                                     
                                                                           
        reachable_accounts = set()
        if policy_df is not None:
            phone_reachable = set(
                policy_df[
                    policy_df["prescribed_action"].isin(
                        [
                            "Keep dialling, at the best time slot",
                            "Switch channel (WhatsApp, field) instead of redialling",
                            "Move to another number on file",
                        ]
                    )
                ]["account_id"].unique()
            )
            reachable_accounts.update(phone_reachable)

                                                                        
        if address_policy_df is not None:
            addr_reachable = set(
                address_policy_df[
                    address_policy_df["prescribed_action"].isin(
                        ["Visit", "Change the visit time"]
                    )
                ]["account_id"].unique()
            )
            reachable_accounts.update(addr_reachable)

        acc_queue["has_reachable_channel"] = acc_queue["account_id"].isin(reachable_accounts)

                                                                                     
                                                       
        eligible_queue = acc_queue[
            (~acc_queue["has_reachable_channel"]) &
            (acc_queue["expected_net_value_trace"] > 0)
        ].sort_values("expected_net_value_trace", ascending=False).reset_index(drop=True)

        return eligible_queue
