"""Validate package integrity, schemas, numerical anchors, and generated outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".md", ".py", ".json", ".csv", ".txt", ".cff", ".gitignore"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    checks: list[dict[str, Any]] = []

    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "DATA_SOURCES.md",
        "DATA_DICTIONARY.md",
        "requirements.txt",
        "configs/endpoint_applicability.json",
        "configs/evaluation_settings.json",
        "configs/expected_summary.json",
        "data/manifest.json",
        "data/input_hashes.json",
        "data/protocol/manifest.json",
        "data/protocol/cards_trace_roles.csv",
        "data/protocol/road_group_inventory.csv",
        "scripts/build_endpoint_contract.py",
        "scripts/aggregate_formal_results.py",
        "scripts/build_paper_outputs.py",
        "scripts/validate_release.py",
        "tests/test_release.py",
        "training/README.md",
        "training/requirements.txt",
        "training/configs/cards_training.json",
        "training/configs/cards_operational.json",
        "training/data/protocol/road_activity_groups.csv",
        "training/data/protocol/ctat_manifest.csv",
        "training/data/protocol/cards_trace_roles.csv",
        "training/data/protocol/manifest.json",
        "training/scripts/road_training.py",
        "training/scripts/road_evaluation.py",
        "training/scripts/ctat_features.py",
        "training/scripts/ctat_models.py",
        "training/scripts/ctat_evaluation.py",
        "training/scripts/cards_features.py",
        "training/scripts/cards_cache.py",
        "training/scripts/cards_profile.py",
        "training/scripts/cards_prepare_cache.py",
        "training/scripts/cards_training.py",
        "training/scripts/cards_models.py",
        "training/scripts/cards_scoring.py",
        "training/scripts/cards_operational.py",
        "training/tests/test_training.py",
    ]
    missing = [path for path in required if not (root / path).is_file()]
    add(checks, "required_files", not missing, "missing=" + ",".join(missing) if missing else f"{len(required)} files present")

    data_dir = root / "data" / "source_data"
    manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    hash_errors: list[str] = []
    for item in manifest["files"]:
        path = data_dir / item["file"]
        if not path.is_file():
            hash_errors.append(f"missing:{item['file']}")
            continue
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            hash_errors.append(f"mismatch:{item['file']}")
    add(checks, "source_data_hashes", not hash_errors, ";".join(hash_errors) if hash_errors else f"{len(manifest['files'])} files verified")

    protocol_dir = root / "data" / "protocol"
    protocol_manifest = json.loads((protocol_dir / "manifest.json").read_text(encoding="utf-8"))
    protocol_hash_errors: list[str] = []
    for item in protocol_manifest["files"]:
        path = protocol_dir / item["file"]
        if not path.is_file():
            protocol_hash_errors.append(f"missing:{item['file']}")
            continue
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            protocol_hash_errors.append(f"mismatch:{item['file']}")
    add(checks, "protocol_hashes", not protocol_hash_errors, ";".join(protocol_hash_errors) if protocol_hash_errors else f"{len(protocol_manifest['files'])} files verified")

    training_protocol_dir = root / "training" / "data" / "protocol"
    training_manifest = json.loads((training_protocol_dir / "manifest.json").read_text(encoding="utf-8"))
    training_protocol_errors: list[str] = []
    for item in training_manifest["files"]:
        path = training_protocol_dir / item["file"]
        if not path.is_file():
            training_protocol_errors.append(f"missing:{item['file']}")
            continue
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            training_protocol_errors.append(f"mismatch:{item['file']}")
            continue
        if len(read_csv(path)) != item["rows"]:
            training_protocol_errors.append(f"rows:{item['file']}")
    add(checks, "training_protocol_files", not training_protocol_errors, ";".join(training_protocol_errors) if training_protocol_errors else f"{len(training_manifest['files'])} files verified")

    expected = json.loads((root / "configs" / "expected_summary.json").read_text(encoding="utf-8"))
    csv_rows = {path.name: read_csv(path) for path in data_dir.glob("*.csv")}
    row_errors = [f"{name}:{len(csv_rows.get(name, []))}!={count}" for name, count in expected["row_counts"].items() if name != "dataset_endpoint_contract.csv" and len(csv_rows.get(name, [])) != count]
    add(checks, "source_row_counts", not row_errors, ";".join(row_errors) if row_errors else "all source row counts match")
    protocol_rows = {path.name: read_csv(path) for path in protocol_dir.glob("*.csv")}
    protocol_row_errors = [
        f"{name}:{len(protocol_rows.get(name, []))}!={count}"
        for name, count in expected["protocol_row_counts"].items()
        if len(protocol_rows.get(name, [])) != count
    ]
    add(checks, "protocol_row_counts", not protocol_row_errors, ";".join(protocol_row_errors) if protocol_row_errors else "all protocol row counts match")

    anchors = expected["numeric_anchors"]
    ctat_total = sum(int(row["unique_microsegments"]) for row in csv_rows["ctat_label_granularity.csv"])
    road_frame = next(row for row in csv_rows["road_rank_reversal.csv"] if row["probe"] == "frame")
    road_timing = next(row for row in csv_rows["road_rank_reversal.csv"] if row["probe"] == "timing")
    cards_primary = next(row for row in csv_rows["cards_operational_sensitivity.csv"] if row["family"] == "HGB" and float(row["budget_fae_h"]) == 30.0 and float(row["cooldown_seconds"]) == 1.0)
    zero_folds = len({int(row["fold"]) for row in csv_rows["cards_mechanism_tpr.csv"] if math.isclose(float(row["mechanism_tpr"]), 0.0, abs_tol=1e-12)})
    anchor_tests = {
        "ctat_total": ctat_total == anchors["ctat_unique_microsegments_total"],
        "road_frame": math.isclose(float(road_frame["frame_macro_f1"]), anchors["road_frame_macro_f1_frame_probe"], abs_tol=1e-12),
        "road_timing": math.isclose(float(road_timing["der_100ms"]), anchors["road_der_100ms_timing_probe"], abs_tol=1e-12),
        "road_rmttd": math.isclose(float(road_timing["rmttd_5s"]), anchors["road_rmttd_5s_timing_probe"], abs_tol=1e-12),
        "cards_primary": math.isclose(float(cards_primary["mechanism_macro_tpr_mean"]), anchors["cards_hgb_primary_macro_tpr_mean"], abs_tol=1e-12),
        "zero_folds": zero_folds == anchors["cards_zero_detection_fold_count"],
    }
    add(checks, "numeric_anchors", all(anchor_tests.values()), json.dumps(anchor_tests, sort_keys=True))

    contract_path = root / "outputs" / "endpoint_contract" / "dataset_endpoint_contract.csv"
    if contract_path.is_file():
        contract = read_csv(contract_path)
        lookup = {(row["dataset"], row["endpoint"]): row["status"] for row in contract}
        contract_tests = {
            "row_count": len(contract) == expected["row_counts"]["dataset_endpoint_contract.csv"],
            "cards_frame_A": lookup.get(("CarDS", "frame_classification")) == "A",
            "ctat_event_NA": lookup.get(("CTAT", "event_response")) == "N/A",
            "hcrl_event_D": lookup.get(("HCRL-CH", "event_response")) == "D",
            "pooling_P": all(row["status"] == "P" for row in contract if row["endpoint"] == "cross_dataset_pooled_score"),
        }
        add(checks, "endpoint_contract", all(contract_tests.values()), json.dumps(contract_tests, sort_keys=True))
    else:
        add(checks, "endpoint_contract", False, "run build_endpoint_contract.py")

    expected_outputs = [root / "outputs" / "paper" / "main_result_table.csv", root / "outputs" / "paper" / "build_manifest.json"]
    for stem in ("figure2_endpoint_applicability_and_rank_divergence", "figure3_cards_operational_sensitivity", "figure4_cards_mechanism_tail_risk"):
        expected_outputs.extend(root / "outputs" / "paper" / "figures" / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg"))
    absent_outputs = [path.relative_to(root).as_posix() for path in expected_outputs if not path.is_file() or path.stat().st_size == 0]
    add(checks, "generated_outputs", not absent_outputs, ",".join(absent_outputs) if absent_outputs else f"{len(expected_outputs)} generated files present")

    stage_pattern = re.compile(r"\bR\d+(?:[._]\d+)+(?:[A-Z])?\b", re.IGNORECASE)
    local_pattern = re.compile(r"\b[A-Za-z]:[\\/](?:Users|CAN)[\\/]", re.IGNORECASE)
    leaks: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or "outputs" in path.relative_to(root).parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if stage_pattern.search(text) or local_pattern.search(text):
            leaks.append(path.relative_to(root).as_posix())
    add(checks, "public_text_scan", not leaks, ",".join(leaks) if leaks else "no internal stage labels or local paths")

    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {"status": status, "checks": checks}
    report_path = root / "outputs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
