import numpy as np
import pandas as pd
from src.config import (
    ACTION_DIAL_BEST_SLOT,
    ACTION_SWITCH_CHANNEL,
    ACTION_RETRY_BACKOFF,
    ACTION_MOVE_NUMBER,
    ACTION_STOP_SUPPRESS,
    ACTION_FPC_THIRD_PARTY,
    ACTION_TRIGGER_TRACE,
)


class PolicyEngine:
    

    def __init__(
        self,
        liveness_dead_threshold=0.25,
        liveness_active_threshold=0.55,
        avoidance_rpc_threshold=0.18,
        switched_off_streak_threshold=3,
    ):
        self.liveness_dead_threshold = liveness_dead_threshold
        self.liveness_active_threshold = liveness_active_threshold
        self.avoidance_rpc_threshold = avoidance_rpc_threshold
        self.switched_off_streak_threshold = switched_off_streak_threshold

    def prescribe_actions(self, df, p_active, p_rpc, account_phones_map=None):
        
        states = []
        actions = []
        reason_codes = []

        for i in range(len(df)):
            row = df.iloc[i]
            pa = p_active[i]
            prpc = p_rpc[i]

            relation = str(row.get("relation_recorded", "self")).lower()
            shared_count = row.get("accounts_sharing_phone", 1)
            wrong_number_count = row.get("prior_wrong_number_count", 0)
            consec_switched_off = row.get("consecutive_switched_off", 0)
            prior_hangup = row.get("prior_customer_hangup_count", 0)
            prior_rejected = row.get("prior_call_rejected_count", 0)
            prior_attempts = row.get("prior_attempts", 0)

                                                                 
            if wrong_number_count > 0 or (shared_count > 2 and "self" not in relation):
                state = "Recycled to a new subscriber"
                action = ACTION_STOP_SUPPRESS
                reason = "Recycled / stranger detected; halt dialling immediately"

                                                            
            elif relation != "self" and "self" not in relation:
                state = "Third party (relative, employer, reference)"
                action = ACTION_FPC_THIRD_PARTY
                reason = "Contact is a reference/relative; enforce Fair Practices Code"

                                                                                         
            elif (consec_switched_off >= self.switched_off_streak_threshold) and (row.get("flag_invalid", 0) == 0):
                state = "Switched off long-term"
                                                                                 
                acc_id = row.get("account_id")
                has_alternate_number = False
                if account_phones_map and acc_id in account_phones_map:
                    available_phones = account_phones_map[acc_id]
                    if len(available_phones) > 1:
                        has_alternate_number = True

                if has_alternate_number:
                    action = ACTION_MOVE_NUMBER
                    reason = f"Persistent switch-off ({consec_switched_off} streak); cascade to alternate phone on file"
                else:
                    action = ACTION_TRIGGER_TRACE
                    reason = f"Persistent switch-off ({consec_switched_off} streak) and no alternate phone; trigger Skip-Trace"

                                                                                                         
            elif pa < self.liveness_dead_threshold or row.get("flag_invalid", 0) == 1:
                state = "Invalid from the start"
                action = ACTION_TRIGGER_TRACE
                reason = f"Line dead / unassigned (P_active={pa:.2f}); escalate to Skip-Trace"

                                        
            elif (consec_switched_off > 0) or (row.get("flag_not_reachable", 0) == 1):
                state = "Temporarily unreachable"
                action = ACTION_RETRY_BACKOFF
                reason = "Transient network unavailability; retry later with backoff"

                                                                   
                                                                                            
            elif (
                (pa >= self.liveness_active_threshold) and
                (
                    (prpc < self.avoidance_rpc_threshold and prior_attempts >= 2) or
                    (prior_hangup >= 1 or prior_rejected >= 1)
                )
            ):
                state = "Valid, but borrower avoiding"
                action = ACTION_SWITCH_CHANNEL
                reason = f"Line alive (P_active={pa:.2f}) but borrower avoiding (P_rpc={prpc:.2f}); switch to WhatsApp/field"

                                    
            elif pa >= self.liveness_active_threshold:
                state = "Valid and reachable"
                action = ACTION_DIAL_BEST_SLOT
                reason = f"High contactability (P_active={pa:.2f}, P_rpc={prpc:.2f}); dial at optimal diurnal window"

            else:
                                  
                state = "Temporarily unreachable"
                action = ACTION_RETRY_BACKOFF
                reason = "Uncertain state; backoff before next contact attempt"

            states.append(state)
            actions.append(action)
            reason_codes.append(reason)

        policy_df = df.copy()
        policy_df["predicted_state"] = states
        policy_df["prescribed_action"] = actions
        policy_df["action_reason"] = reason_codes
        policy_df["prob_active"] = p_active
        policy_df["prob_rpc"] = p_rpc

        return policy_df
