Training and evaluation
=======================

The scripts in this directory reproduce the data preparation, probe training, prediction generation, and operational evaluation used in the study. They cover the ROAD, CTAT, and CarDS workflows. The analysis scripts in the repository root rebuild the endpoint contract, main result table, and principal figures from the resulting source tables.

Environment
-----------

Python 3.12 is recommended.

```bash
python -m venv .venv
python -m pip install -r training/requirements.txt
```

Dataset placement
-----------------

Download the datasets from the locations listed in `../DATA_SOURCES.md` and place the archives as follows:

```text
training/data/raw/road.zip
training/data/raw/cantrainandtest.zip
training/data/raw/CarDS.zip
```

Run the following commands from the `training` directory.

ROAD
----

```bash
python scripts/road_training.py --mode smoke --out work/road_smoke
python scripts/road_evaluation.py --mode formal --out work/road_formal
```

The ROAD workflow builds frame, timing, and context probes under group-disjoint evaluation and writes frame classification, attack-event response, and normal-traffic alarm tables.

CTAT
----

```bash
python scripts/ctat_evaluation.py --mode check
python scripts/ctat_evaluation.py --mode formal --out work/ctat_formal
```

The CTAT workflow trains the frame, timing, and context probes on each official training set and writes frame, positive-segment, and independent-normal evaluation tables. The paper uses the positive-segment table to assess label granularity and event-endpoint eligibility.

CarDS
-----

```bash
python scripts/cards_prepare_cache.py --config configs/cards_training.json --out work/cards_cache_receipt --build
python scripts/cards_features.py --config configs/cards_training.json --out work/cards_smoke --smoke
python scripts/cards_operational.py --config configs/cards_operational.json --out work/cards_check --check
python scripts/cards_operational.py --config configs/cards_operational.json --out work/cards_formal --formal
```

The CarDS workflow uses the materialized three-fold mechanism-disjoint roles in `data/protocol/cards_trace_roles.csv`. It constructs causal identity, timing, payload, and context features. Logistic regression and histogram-gradient boosting use the 21 features of the current frame. The early-fusion TCN uses the full causal sequence of 16 same-bus frames, with 21 features per frame. Thresholds are calibrated on normal traffic, and the workflow writes mechanism coverage, deadline detection, restricted detection time, and false-alarm-event results.

Tests
-----

```bash
python -m unittest discover -s training/tests -v
```
