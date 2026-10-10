import os
import numpy as np
import pandas as pd
from src.config import (
    ADDRESSES_PATH,
    FIELD_VISITS_PATH,
    ACTION_ADDR_VISIT,
    ACTION_ADDR_CHANGE_TIME,
    ACTION_ADDR_TRACE_NEW,
    ACTION_ADDR_RESOLVE_LOC,
    ACTION_ADDR_FABRICATED,
)


class FieldPolicyEngine:
    """
    Evaluates address health and prescribes field collection actions mandated by
    Problem Statement 2 (Addresses table, page 5).
    """

    def __init__(self):
        pass

    def evaluate_addresses(self, addresses_df=None, field_visits_df=None):
        if addresses_df is None:
            addresses_df = pd.read_csv(ADDRESSES_PATH)
        if field_visits_df is None:
            field_visits_df = pd.read_csv(FIELD_VISITS_PATH)

        # Summarize past field visits per address_id
        fv_summary = (
            field_visits_df.sort_values(["address_id", "start_ts"])
            .groupby("address_id")
            .agg(
                visit_count=("visit_id", "count"),
                last_outcome=("outcome", "last"),
                last_dwell_s=("dwell_s", "last"),
                met_borrower_count=("outcome", lambda s: (s == "met_borrower").sum()),
                met_family_count=("outcome", lambda s: (s == "met_family").sum()),
                locked_premises_count=("outcome", lambda s: (s == "locked_premises").sum()),
                shifted_count=("outcome", lambda s: (s == "neighbour_says_shifted").sum()),
                no_person_count=("outcome", lambda s: (s == "no_such_person").sum()),
                not_traceable_count=("outcome", lambda s: (s == "address_not_traceable").sum()),
                cash_collected_count=("outcome", lambda s: (s == "cash_collected").sum()),
            )
            .reset_index()
        )

        merged = addresses_df.merge(fv_summary, on="address_id", how="left")
        merged["visit_count"] = merged["visit_count"].fillna(0).astype(int)
        merged["last_dwell_s"] = merged["last_dwell_s"].fillna(0)
        merged["address_text_len"] = merged["address_text"].astype(str).str.len()

        states = []
        actions = []
        reasons = []

        for i in range(len(merged)):
            row = merged.iloc[i]
            v_count = row["visit_count"]
            last_out = row["last_outcome"]
            dwell = row["last_dwell_s"]
            text_len = row["address_text_len"]

            if v_count == 0:
                # Never tested address
                if text_len < 18:
                    state = "Fabricated or incomplete at origination"
                    action = ACTION_ADDR_FABRICATED
                    reason = f"Address text incomplete/vague ({text_len} chars); flag to origination & trace"
                else:
                    state = "Valid and occupied"
                    action = ACTION_ADDR_VISIT
                    reason = "Unvisited address with complete description; schedule initial field visit"

            else:
                # Evaluated based on historical field visits
                if last_out in ["met_borrower", "cash_collected"] or row["met_borrower_count"] > 0:
                    state = "Valid and occupied"
                    action = ACTION_ADDR_VISIT
                    reason = "Borrower confirmed at address on prior visit; schedule visit"

                elif last_out in ["locked_premises", "met_family"] or (row["locked_premises_count"] > 0):
                    state = "Valid, but borrower usually absent"
                    action = ACTION_ADDR_CHANGE_TIME
                    reason = "Premises locked or met family; shift visit to evening/weekend slot"

                elif last_out in ["neighbour_says_shifted", "no_such_person"] or (row["shifted_count"] > 0):
                    state = "Borrower has moved"
                    action = ACTION_ADDR_TRACE_NEW
                    reason = "Neighbours or occupants confirmed borrower relocated; trace new address"

                elif last_out == "address_not_traceable":
                    if text_len >= 25 or dwell >= 180:
                        state = "Hard to find on the ground"
                        action = ACTION_ADDR_RESOLVE_LOC
                        reason = f"Detailed address but agent could not locate pin ({dwell:.0f}s dwell); resolve geocode (PS3)"
                    else:
                        state = "Fabricated or incomplete at origination"
                        action = ACTION_ADDR_FABRICATED
                        reason = "Address not traceable and description incomplete; flag to origination team"

                else:
                    state = "Valid and occupied"
                    action = ACTION_ADDR_VISIT
                    reason = "Standard visit schedule"

            states.append(state)
            actions.append(action)
            reasons.append(reason)

        merged["predicted_state"] = states
        merged["prescribed_action"] = actions
        merged["action_reason"] = reasons

        return merged
