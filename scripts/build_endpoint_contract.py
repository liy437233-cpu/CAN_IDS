"""Build the dataset–endpoint applicability contract from metadata profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


ENDPOINTS = (
    "frame_classification",
    "event_response",
    "independent_normal_alarm_burden",
    "mechanism_tail",
    "cross_dataset_pooled_score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_frame(profile: dict[str, Any]) -> tuple[str, str, list[str]]:
    label_type = profile["frame_labels"]["type"]
    if label_type == "exact_publisher":
        return "A", "F01_EXACT_PUBLISHER_FRAME_LABEL", []
    if label_type == "proxy_reconstructed":
        return "D", "F02_RECONSTRUCTED_FRAME_PROXY", ["FRAME_LABEL_PROXY"]
    return "N/A", "F03_NO_FRAME_TARGET_LABEL", ["FRAME_ENDPOINT_UNSUPPORTED"]


def classify_event(profile: dict[str, Any]) -> tuple[str, str, list[str]]:
    event = profile["event_metadata"]
    warnings: list[str] = []
    if event["explicit_intervals"]:
        if not event["physical_command_time"]:
            warnings.append("ANCHOR_NOT_PHYSICAL_COMMAND_TIME")
        return "A", "E01_EXPLICIT_EVENT_INTERVAL", warnings
    if event["label_anchored_start"] and not event["grouping_requires_heuristic"]:
        if not event["physical_command_time"]:
            warnings.append("LABEL_ANCHORED_RESPONSE_ONLY")
        return "A", "E02_UNAMBIGUOUS_LABEL_ANCHOR", warnings
    if event["label_anchored_start"] and event["grouping_requires_heuristic"]:
        warnings.extend(("HEURISTIC_EVENT_GROUPING", "LABEL_ANCHORED_RESPONSE_ONLY"))
        return "D", "E03_HEURISTIC_EVENT_RECONSTRUCTION", warnings
    return "N/A", "E04_NO_INTERPRETABLE_EVENT_ANCHOR", ["EVENT_ENDPOINT_UNSUPPORTED"]


def classify_normal(profile: dict[str, Any]) -> tuple[str, str, list[str]]:
    normal = profile["normal_evaluation"]
    warnings: list[str] = []
    if normal["separate_records"] and normal["duration_available"]:
        if normal["context_shift_warning"]:
            warnings.append("NORMAL_ATTACK_CONTEXT_SHIFT")
        return "A", "N01_SEPARATE_DURATION_KNOWN_NORMAL", warnings
    if normal["duration_available"]:
        return "D", "N02_NONINDEPENDENT_NORMAL_EXPOSURE", ["NORMAL_NOT_INDEPENDENT"]
    return "N/A", "N03_NO_DURATION_KNOWN_NORMAL", ["FAE_H_UNSUPPORTED"]


def classify_mechanism(profile: dict[str, Any]) -> tuple[str, str, list[str]]:
    mechanism = profile["mechanism_evaluation"]
    if mechanism["labels_available"] and mechanism["group_count"] >= 4 and mechanism["mechanism_disjoint_holdout"]:
        return "A", "M01_MECHANISM_DISJOINT_TAIL_SUPPORTED", []
    if mechanism["labels_available"] and mechanism["group_count"] >= 2:
        return "D", "M02_MECHANISM_STRATIFICATION_ONLY", ["NO_MECHANISM_DISJOINT_HOLDOUT"]
    return "N/A", "M03_NO_MECHANISM_GROUPS", ["MECHANISM_TAIL_UNSUPPORTED"]


def classify_pooling(profile: dict[str, Any]) -> tuple[str, str, list[str]]:
    pooling = profile["cross_dataset_pooling"]
    if pooling["common_label_semantics"] and pooling["common_probe_representation"]:
        return "A", "P01_HARMONIZED_POOLING_CONTRACT", []
    return "P", "P02_NONHARMONIZED_CROSS_DATASET_POOLING", ["CROSS_DATASET_POOLING_PROHIBITED"]


CLASSIFIERS: dict[str, Callable[[dict[str, Any]], tuple[str, str, list[str]]]] = {
    "frame_classification": classify_frame,
    "event_response": classify_event,
    "independent_normal_alarm_burden": classify_normal,
    "mechanism_tail": classify_mechanism,
    "cross_dataset_pooled_score": classify_pooling,
}


def evaluate(profile: dict[str, Any]) -> list[dict[str, Any]]:
    shared = ["SINGLE_VEHICLE_EXTERNAL_VALIDITY"] if profile["external_validity"]["single_vehicle"] else []
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        status, reason_code, warnings = CLASSIFIERS[endpoint](profile)
        rows.append(
            {
                "dataset": profile["dataset_id"],
                "endpoint": endpoint,
                "status": status,
                "reason_code": reason_code,
                "warnings": sorted(set(shared + warnings)),
                "used_to_design_rules": bool(profile["used_to_design_rules"]),
                "design_role": profile["design_role"],
                "paper_reporting_role": profile["paper_reporting_role"][endpoint],
            }
        )
    return rows


def write_contract(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "dataset",
        "endpoint",
        "status",
        "reason_code",
        "warnings",
        "used_to_design_rules",
        "design_role",
        "paper_reporting_role",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "warnings": "|".join(row["warnings"])})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    mismatches: list[dict[str, str]] = []
    for profile in config["datasets"]:
        decisions = evaluate(profile)
        rows.extend(decisions)
        traces[profile["dataset_id"]] = {
            "evidence": profile["evidence"],
            "input_profile": profile,
            "decisions": decisions,
        }
        observed = {row["endpoint"]: row["status"] for row in decisions}
        for endpoint, expected in profile["expected_contract"].items():
            if observed.get(endpoint) != expected:
                mismatches.append(
                    {"dataset": profile["dataset_id"], "endpoint": endpoint, "expected": expected, "observed": str(observed.get(endpoint))}
                )

    external_count = sum(not bool(item["used_to_design_rules"]) for item in config["datasets"])
    status = "PASS" if not mismatches and external_count > 0 else "FAIL"
    write_contract(output / "dataset_endpoint_contract.csv", rows)
    (output / "decision_traces.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": status,
        "contract_id": config["contract_id"],
        "config_file": args.config.as_posix(),
        "config_sha256": sha256(config_path),
        "dataset_count": len(config["datasets"]),
        "design_dataset_count": len(config["datasets"]) - external_count,
        "external_exercise_dataset_count": external_count,
        "endpoint_decision_count": len(rows),
        "mismatches": mismatches,
        "interpretation_boundary": "Deterministic replay verifies the encoded metadata and rules; it does not establish inter-rater agreement for metadata encoding.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
