# CreditNirvana PS2: Right-Party Contact & Skip-Trace Optimizer

This repository implements a data-driven collections strategy that optimizes outbound dialing decisions for debt recovery while respecting compliance constraints. It combines feature engineering, calibrated predictive models, a state-to-action policy engine, and a value-of-information skip-trace prioritization layer.

The project is designed around the problem statement for a debt collections contact optimization challenge: identify valid contact points, reduce wasted calls on invalid numbers, increase right-party contact (RPC), and minimize third-party disclosure risk.

## What this project does

- Predicts line liveness / connectivity for each attempted contact number
- Predicts the probability of right-party contact (RPC) for an active line
- Identifies likely avoidance / invalid / recycled contact situations
- Maps predicted states to operational actions such as:
  - continue dialing at the best slot
  - switch channels (WhatsApp / field visit)
  - trigger skip-trace
  - suppress contact to avoid disclosure risk
- Prioritizes skip-trace candidates using an expected net value model
- Produces metrics and plots for benchmarking performance against incumbent policies

## Repository structure

- `src/`
  - `config.py` — project paths and action constants
  - `feature_engineering.py` — telephony, account, and cross-channel feature generation
  - `models.py` — LightGBM-based liveness and RPC models with calibration
  - `policy_engine.py` — state-to-action rules based on predicted probabilities
  - `skip_trace_optimizer.py` — expected-value optimization for skip-trace queue ranking
- `run_pipeline.py` — end-to-end training, prediction, and artifact generation
- `generate_metrics_and_plots.py` — computes official metrics and saves plots
- `check_validation.py` — validation summary for model calibration and performance
- `shared/` — source data tables shared by the challenge
- `output/` — generated model outputs, policy predictions, metrics, and plots
- `phones.csv` — phone-level metadata and linkage information
- `skip_traces.csv` — historical skip-trace outcomes
- `verified_contact_points.csv` — audited contact-point ground truth

## Core workflow

1. Load raw account, phone, dial-attempt, payment, and field-visit datasets.
2. Compute strict pre-attempt historical features without leakage.
3. Train a dual-model suite:
   - line liveness model: `P(active line)`
   - RPC model: `P(RPC | active line)`
4. Apply the policy engine to convert model scores into operational actions.
5. Rank skip-trace targets based on expected net recovery value.
6. Save outputs and evaluate performance against expected metrics.

## Setup

This project uses Python 3.10+ and the following packages:

- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn

Install dependencies with:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

## Run the pipeline

From the repository root:

```bash
python run_pipeline.py
```

This will:

- build the engineered feature dataset
- train the liveness and RPC models
- apply the policy engine to the test set
- validate results against `verified_contact_points.csv`
- generate `test_predictions_with_policy.csv`
- generate `skip_trace_priority_queue.csv`

## Generate official metrics and plots

```bash
python generate_metrics_and_plots.py
```

This produces the official evaluation outputs, including:

- AUC and Brier scores
- RPC rate per 1,000 dial attempts
- invalid-number wastage reduction metrics
- skip-trace hit-rate estimates
- calibration and performance plots saved under `output/plots/`

## Validate the model suite

```bash
python check_validation.py
```

This script checks the train/validation/test split setup and reports validation accuracy and calibration metrics for the liveness and RPC models.

## Output artifacts

The `output/` directory contains the generated analysis outputs, such as:

- `test_predictions_with_policy.csv`
- `skip_trace_priority_queue.csv`
- `verified_audit_comparison.csv`
- `plots/` with performance charts and calibration diagnostics

## Notes

- The project uses a strict no-leakage feature pipeline, where features are built only from historical data prior to the current attempt.
- The policy layer is rule-based and designed to reflect regulatory and operational constraints rather than pure probability maximization alone.
- Skip-trace prioritization combines probability of a successful trace with expected recovery value and the cost of a trace.

## License

This project does not currently include a project-level license file. Please check with the repository owner or the challenge instructions before redistributing or using the code commercially.

## Contact

For questions or collaboration, contact the repo owner or the project maintainer associated with this repository.
