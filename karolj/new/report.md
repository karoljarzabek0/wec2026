# Fresh Modeling Iteration Report (karolj/new)

## 1. Naive Baseline (01_baseline_ml.py)
*   **ROC-AUC:** 0.6423
*   **PR-AUC:** 0.0541
*   **SHAP Top Feature:** `jersey_number` (Proxying for position)
*   **Diagnosis:** The model was distracted by proxy features and likely contained future leakage through `minute_out` and `subbed`.

## 2. Cleaned Model (02_clean_ml.py)
*   **Changes:** Removed `minute_out`, `subbed`, and `jersey_number`. Added `mins_on_pitch` and team-level aggregates (`team_last15_shots`).
*   **ROC-AUC:** 0.6988 (+0.05 increase)
*   **PR-AUC:** 0.0765 (+41% increase)
*   **Diagnosis:** Removing noisy identifiers and fixing the time-on-pitch logic significantly improved the model's ability to generalize.

## 3. Tuned Model (03_tuned_ml.py)
*   **Changes:** Applied Optuna hyperparameter optimization to the Cleaned Model.
*   **ROC-AUC:** 0.6984 (Stable)
*   **PR-AUC:** 0.0852 (**+57% increase from naive baseline**)
*   **Best Parameters:** `learning_rate: 0.10`, `max_depth: 8`, `n_estimators: 149`.

## Honest Assessment
The jump in ROC-AUC from 0.64 to nearly 0.70 is substantial. By focusing on **physical capacity** (`cumul_peak_speed`) and **fatigue** (`mins_on_pitch`), the model has found a much more reliable signal than the original "identifier-heavy" model.

The fact that `position_D` (Defenders) is a top SHAP feature (negative impact) confirms the model is correctly anchoring its predictions on attacking threats.

## Next Steps for "Even Better"
To break the 0.70 ROC-AUC barrier, we should:
1.  **Re-integrate Shot Quality:** Bring back the `l15_shot_volley_share` which showed promise in the first phase.
2.  **Pressure Resilience:** Re-integrate `last15_press_turnover_vol` as a proxy for creative risk-taking.
3.  **Interaction Effects:** Specifically, the interaction between `mins_on_pitch` and `last15_distance` (efficiency decay).
