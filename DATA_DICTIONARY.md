Data dictionary
===============

`ctat_label_granularity.csv`
----------------------------

- `task`: official CTAT task identifier.
- `unique_microsegments`: number of contiguous positive-label segments.
- `median_frames`: median frames per positive segment.
- `median_duration_ms`: median elapsed duration per positive segment in milliseconds.
- `single_frame_fraction`: fraction of positive segments containing one frame.
- `duration_le_10ms_fraction`: fraction of positive segments lasting no more than 10 ms.

`road_rank_reversal.csv`
------------------------

- `probe`: probe family (`frame`, `context`, or `timing`).
- `frame_macro_f1`: frame-level macro-averaged F1.
- `der_100ms`: deadline event detection rate at 100 ms.
- `rmttd_5s`: restricted mean time to detection at a 5 s horizon. Detected events contribute `min(delay, 5 s)` and undetected events contribute 5 s.
- `normal_fae_h`: false alarm events per hour on independent normal captures; lower is better.
- `frame_rank`, `der_rank`, `fae_rank`: dense within-dataset ranks.

`cards_operational_sensitivity.csv`
-----------------------------------

- `family`: detector family (`HGB`, `TCN`, or `LR`).
- `budget_fae_h`: calibration alarm budget in false alarm events per hour.
- `cooldown_seconds`: alarm-merging cooldown in seconds.
- `observation_count`: number of fold/seed observations summarized in the row.
- `independence_note`: interpretation of the observation count.
- `*_mean`, `*_min`, `*_max`: mean and range for mechanism macro-TPR, DER@100ms, outer-normal FAE/h, and restricted mean time to detection at a 5 s horizon.

`cards_mechanism_tpr.csv`
-------------------------

- `fold`: mechanism-disjoint outer-fold index.
- `mechanism`: attack mechanism identifier.
- `rank_within_fold`: ascending rank by mechanism TPR.
- `mechanism_tpr`: true-positive rate for the mechanism.
- `mechanism_macro_tpr`: unweighted mean TPR across mechanisms in the fold.
- `lower_quartile_tpr`: lower-quartile mechanism TPR.
- `worst_mechanism_tpr`: minimum mechanism TPR.
- `mechanism_count`: number of held-out mechanisms in the fold.
- `der_100ms`: trace-mean deadline event detection rate at 100 ms.
- `rmttd_5s`: restricted mean time to detection with a 5 s horizon.
- `outer_normal_fae_h`: false alarm events per hour on the outer normal split.

Endpoint contract
-----------------

- `dataset`: dataset identifier.
- `endpoint`: endpoint family evaluated by the rules.
- `status`: `A`, `D`, `N/A`, or `P`.
- `reason_code`: deterministic reason code for the decision.
- `warnings`: pipe-delimited interpretation limits.
- `used_to_design_rules`: whether the dataset informed rule design.
- `design_role`: role assigned to the dataset metadata.
- `paper_reporting_role`: how the endpoint is used in the study.

`cards_trace_roles.csv`
-----------------------

- `outer_fold`: CarDS mechanism-disjoint outer-fold index.
- `trace_name`: publisher trace filename.
- `family`: attack family or benign group.
- `attack_mechanism_group`: mechanism used to construct disjoint outer folds.
- `role`: training, calibration, or outer-test role.
- `is_outer_test_attack`: whether the trace is an attack test trace for the fold.
- `r_t_label_is_model_input`: confirms that the publisher R/T label is not a model input.
- `filename_attack_semantics_is_model_input`: confirms that filename semantics are not a model input.

`road_group_inventory.csv`
--------------------------

- `model`: ROAD probe family.
- `heldout_group`: activity group excluded from that fit.
- `kind`: attack or normal group.
- `positive_frames`: positive-frame count in the fit inventory.
- `negative_frames`: negative-frame count in the fit inventory.
