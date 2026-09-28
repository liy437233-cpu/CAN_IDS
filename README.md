Operational Evaluation of CAN Intrusion Detection
=================================================

This repository contains the analysis and training code used to evaluate whether public Controller Area Network (CAN) intrusion datasets support frame-level, event-level, mechanism-tail, and normal-operation alarm endpoints.

The package reproduces:

1. a machine-readable dataset-endpoint applicability contract;
2. the main result table;
3. the three principal figures used in the accompanying study;
4. the ROAD, CTAT, and CarDS data preparation, probe-training, prediction, and operational-evaluation workflows.

Repository layout
-----------------

```text
configs/                    Dataset metadata used by the applicability rules
data/source_data/           Aggregated source values for tables and figures
data/protocol/              Materialized split roles and group inventories
scripts/                    Contract, figure, and validation programs
tests/                      Automated integrity tests
outputs/                    Generated files (not required as input)
training/                   Dataset preparation, probe training, and evaluation
DATA_SOURCES.md             Dataset identifiers and acquisition links
DATA_DICTIONARY.md          Definitions for the distributed CSV columns
```

Environment
-----------

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r training/requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r training/requirements.txt
```

Reproduce the outputs
---------------------

Run the commands from the repository root.

```bash
python scripts/build_endpoint_contract.py --config configs/endpoint_applicability.json --output outputs/endpoint_contract
python scripts/build_paper_outputs.py --data-dir data/source_data --contract outputs/endpoint_contract/dataset_endpoint_contract.csv --output outputs/paper
python scripts/validate_release.py --package-root .
python -m unittest discover -s tests -v
```

The first command writes the endpoint contract and decision traces. The second command writes the main table, PNG/PDF/SVG figures, and a machine-readable build manifest. The validation command checks required files, input hashes, schemas, row counts, selected numerical anchors, generated outputs, and local-path leakage.

Aggregate completed experiment runs
-----------------------------------

After the three formal workflows have completed, rebuild the source tables before generating the paper outputs:

```bash
python scripts/aggregate_formal_results.py --road-results training/work/road_formal --ctat-results training/work/ctat_formal --cards-results training/work/cards_formal --output data/source_data --manifest data/manifest.json
```

This step calculates ROAD RMTTD at a 5 s horizon from the event rows, summarizes CTAT label microsegments, aggregates the CarDS folds and sensitivity grid, and writes the source-data manifest.

Reproduce training and evaluation
---------------------------------

The ROAD, CTAT, and CarDS workflows are documented in `training/README.md`. They use the materialized split roles, feature settings, model parameters, random seeds, threshold rules, and operational-metric definitions applied in the study. The resulting formal artifacts are the inputs to `scripts/aggregate_formal_results.py`.

Endpoint statuses
-----------------

- `A`: semantically admissible as a formal within-dataset endpoint.
- `D`: descriptive or proxy use only.
- `N/A`: unsupported by the available label semantics or metadata.
- `P`: prohibited for pooled cross-dataset performance inference.

`paper_reporting_role` is a separate field. It records whether an admissible endpoint is primary, secondary, excluded, or used only for the external metadata exercise.

Data and code availability
--------------------------

The repository includes the aggregated source tables, materialized split roles, evaluation settings, analysis scripts, and training workflows used for the reported results. The public CAN datasets are available from the repositories listed in `DATA_SOURCES.md`.

The validation command checks release integrity and consistency with the expected outputs. It does not independently establish the scientific conclusions.

Citation
--------

Use the metadata in `CITATION.cff`. Cite the original dataset publications and repository records when using CTAT, ROAD, CarDS, or HCRL Car-Hacking data.
