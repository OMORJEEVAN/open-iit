import os

                  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DIR = os.path.join(BASE_DIR, "shared")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

               
ACCOUNTS_PATH = os.path.join(SHARED_DIR, "accounts.csv")
ADDRESSES_PATH = os.path.join(SHARED_DIR, "addresses.csv")
AGENTS_PATH = os.path.join(SHARED_DIR, "agents.csv")
DIAL_ATTEMPTS_PATH = os.path.join(SHARED_DIR, "dial_attempts.csv")
FIELD_VISITS_PATH = os.path.join(SHARED_DIR, "field_visits.csv")
LENDERS_PATH = os.path.join(SHARED_DIR, "lenders.csv")
PAYMENTS_PATH = os.path.join(SHARED_DIR, "payments.csv")
SPLITS_PATH = os.path.join(SHARED_DIR, "splits.csv")

PHONES_PATH = os.path.join(BASE_DIR, "phones.csv")
SKIP_TRACES_PATH = os.path.join(BASE_DIR, "skip_traces.csv")
VERIFIED_CONTACTS_PATH = os.path.join(BASE_DIR, "verified_contact_points.csv")

                      
COST_TELE_CALLER = 20.0
COST_VOICE_BOT = 2.0
COST_FIELD_VISIT = 250.0
COST_SKIP_TRACE = 89.0

                                                               
RPC_DISPOSITIONS = [
    "rpc_ptp",
    "rpc_call_back",
    "rpc_hung_up",
    "rpc_refused",
    "rpc_hardship",
    "rpc_dispute",
    "rpc_claims_paid",
]

THIRD_PARTY_DISPOSITIONS = [
    "third_party_contact",
    "third_party_ptp",
]

AVOIDANCE_DISPOSITIONS = [
    "call_rejected",
]

DEAD_DISPOSITIONS = [
    "invalid_number",
]

NETWORK_ACTIVE_RESPONSES = [
    "answered",
    "busy_rejected",
    "ring_no_answer",
]

NETWORK_DEAD_RESPONSES = [
    "number_does_not_exist",
]

                                
ACTION_DIAL_BEST_SLOT = "Keep dialling, at the best time slot"
ACTION_SWITCH_CHANNEL = "Switch channel (WhatsApp, field) instead of redialling"
ACTION_RETRY_BACKOFF = "Retry later, with backoff"
ACTION_MOVE_NUMBER = "Move to another number on file"
ACTION_STOP_SUPPRESS = "Stop at once; suppress to avoid third-party disclosure"
ACTION_FPC_THIRD_PARTY = "Use only within Fair Practices Code rules, never discuss debt"
ACTION_TRIGGER_TRACE = "Trigger Skip-Trace"

                                                                             
ACTION_ADDR_VISIT = "Visit"
ACTION_ADDR_CHANGE_TIME = "Change the visit time"
ACTION_ADDR_TRACE_NEW = "Trace the new address"
ACTION_ADDR_RESOLVE_LOC = "Resolve the location (Problem Statement 3) before writing it off"
ACTION_ADDR_FABRICATED = "Trace, and flag to the origination team"
