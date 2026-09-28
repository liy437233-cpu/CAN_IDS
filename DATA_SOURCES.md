Data sources
============

The repository distributes aggregated analysis values, not raw third-party CAN traffic. Obtain raw data from the original records below and follow the terms stated by each repository.

| Short name | Dataset | Stable source | Role in this analysis |
|---|---|---|---|
| CTAT | can-train-and-test | DTU Data, [10.11583/DTU.24805533](https://doi.org/10.11583/DTU.24805533) | Frame-label granularity and independent-normal evaluation |
| ROAD | Real ORNL Automotive Dynamometer CAN Intrusion Dataset | Zenodo, [10.5281/zenodo.10462796](https://doi.org/10.5281/zenodo.10462796) | Event-interval evaluation and within-dataset ranking comparison |
| CarDS | Controller Area Network and Automotive Ethernet Realistic Data Set | TUdatalib, [10.48328/tudatalib-2179](https://doi.org/10.48328/tudatalib-2179) | Label-anchored event response, mechanism-disjoint evaluation, and alarm-rule sensitivity |
| HCRL-CH | HCRL Car-Hacking Dataset | [Official dataset page](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) | External metadata exercise for the endpoint-applicability rules; no performance result is reported |

Distributed analysis data
-------------------------

`data/source_data/` contains values aggregated from the study's completed evaluation runs. These files support the main table and figures without exposing raw CAN frames or redistributing third-party archives.

- `ctat_label_granularity.csv`: positive-label microsegment counts and duration summaries.
- `road_rank_reversal.csv`: probe-level frame, event-response, and normal-alarm results.
- `cards_operational_sensitivity.csv`: detector-family summaries across alarm budgets and cooldowns.
- `cards_mechanism_tpr.csv`: per-mechanism true-positive rates for three mechanism-disjoint folds.
- `main_result_table.csv`: manuscript-ready summary rows.

The SHA-256 inventory in `data/manifest.json` identifies the exact distributed files.

Split and input receipts
------------------------

- `data/protocol/cards_trace_roles.csv` materializes the three mechanism-disjoint CarDS outer folds and the role assigned to every trace.
- `data/protocol/road_group_inventory.csv` records the held-out ROAD activity group and frame counts for each probe fit.
- `configs/evaluation_settings.json` records the probe settings, alarm grid, random seeds, feature groups, and sampling limits used to create the distributed summaries.
- `data/input_hashes.json` records the raw archive and cached payload hashes used by the completed evaluation.
