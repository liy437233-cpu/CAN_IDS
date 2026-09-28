"""Deterministic CarDS training-input assembly."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import cards_cache as cache_io
import cards_profile as profile_io
import cards_features as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cards_training.json"
CACHE = ROOT / "work" / "cards_cache"
WORKERS = 6


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["family"], row["trace_name"], row["role"], row.get("attack_mechanism_group", ""))


def roles_for_fold(config: dict[str, Any], fold: int) -> dict[str, list[dict[str, str]]]:
    with Path(config["data"]["role_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["outer_fold"]) == fold and row["family"] not in {"uds", "ethernet"}]
    normals = [row for row in rows if row["role"] == "model_train_normal"]
    assignment = config["normal_reference_split"]["trace_assignment"][str(fold)]
    by_name = {row["trace_name"]: row for row in normals}
    expected = set(assignment["reference_traces"]) | set(assignment["normal_training_traces"])
    if expected != set(by_name):
        raise RuntimeError(f"fold {fold} normal-trace assignment does not match the role table")
    result = {"references": [by_name[name] for name in assignment["reference_traces"]], "normal_train": [by_name[name] for name in assignment["normal_training_traces"]], "attack_train": [row for row in rows if row["role"] == "model_train_attack"], "normal_calibration": [row for row in rows if row["role"] == "normal_calibration"], "outer_attack": [row for row in rows if row["role"] == "outer_test_attack"], "outer_normal": [row for row in rows if row["role"] == "outer_test_normal"]}
    if (len(result["references"]), len(result["normal_train"]), len(result["normal_calibration"]), len(result["outer_normal"])) != (10, 9, 7, 13):
        raise RuntimeError(f"fold {fold} normal-role contract failed")
    train_mechanisms = {row["attack_mechanism_group"] for row in result["attack_train"]}
    test_mechanisms = {row["attack_mechanism_group"] for row in result["outer_attack"]}
    if train_mechanisms & test_mechanisms:
        raise RuntimeError(f"fold {fold} attack mechanisms overlap")
    return result


def task_result(cache: Path, row: dict[str, str], profile: dict[str, Any], center: np.ndarray, scale: np.ndarray, group: int, normal_cap: int, attack_cap: int) -> dict[str, Any]:
    return cache_io.worker_sample(str(cache), row, profile, center, scale, 16, group, normal_cap, attack_cap)


def evenly_allocate(capacity: dict[str, int], target: int, seed: str) -> dict[str, int]:
    """Equal trace quota with deterministic redistribution of unused quota."""
    keys = sorted(capacity, key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
    if target > sum(capacity.values()):
        raise RuntimeError("requested more windows than available; no duplication is permitted")
    allocation = {key: min(capacity[key], target // len(keys)) for key in keys}
    remaining = target - sum(allocation.values())
    while remaining:
        progressed = False
        for key in keys:
            if allocation[key] < capacity[key]:
                allocation[key] += 1; remaining -= 1; progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("deterministic redistribution exhausted unexpectedly")
    return allocation


def canonical_arrays(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(items, key=lambda item: canonical_key(item["row"]))
    samples = [sample for item in ordered for sample in item["samples"]]
    if not samples:
        raise RuntimeError("training assembly retained no samples")
    x = np.stack([sample[0] for sample in samples]).astype("<f4", copy=False)
    y = np.asarray([sample[1] for sample in samples], dtype="<i8")
    group = np.asarray([sample[2] for sample in samples], dtype="<i8")
    return x, y, group


def assemble_formal_fold(config: dict[str, Any], fold: int) -> dict[str, Any]:
    """Formal-only assembly: reads cached CAN logs; no test trace is supplied."""
    roles = roles_for_fold(config, fold)
    cache_manifest = json.loads((CACHE / "cache_manifest.json").read_text(encoding="utf-8"))
    needed = sorted({row["family"] for key in ("references", "normal_train", "attack_train") for row in roles[key]})
    if cache_manifest.get("families") != needed:
        raise RuntimeError("full CAN cache does not exactly cover training-side families")
    profile = profile_io.build_cached_profile(CACHE, roles["references"])
    center, scale = profile_io.fit_cached_normalizer(CACHE, roles["references"], profile)
    by_mechanism: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in roles["attack_train"]:
        by_mechanism[row["attack_mechanism_group"]].append(row)
    mechanism_names = sorted(by_mechanism)
    preliminary: list[dict[str, Any]] = []
    work_items = [(row, mechanism_names.index(mechanism)) for mechanism, rows in by_mechanism.items() for row in rows]
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(task_result, CACHE, row, profile, center, scale, group, 0, int(config["sampling"]["positive_trace_cap"])) for row, group in work_items]
        for future in as_completed(futures):
            preliminary.append(future.result())
    selected_attack: list[dict[str, Any]] = []
    cap_mechanism = int(config["sampling"]["positive_window_cap_per_mechanism"])
    for mechanism in mechanism_names:
        rows = [item for item in preliminary if item["row"]["attack_mechanism_group"] == mechanism]
        capacities = {item["row"]["trace_name"]: len(item["samples"]) for item in rows}
        allocation = evenly_allocate(capacities, min(cap_mechanism, sum(capacities.values())), f"cards|fold={fold}|mechanism={mechanism}")
        for item in rows:
            count = allocation[item["row"]["trace_name"]]
            item["samples"] = item["samples"][:count]
            item["counts"]["positive_selected_after_mechanism_balance"] = count
            selected_attack.append(item)
    attack_count = sum(len(item["samples"]) for item in selected_attack)
    normal_capacity = evenly_allocate({row["trace_name"]: attack_count for row in roles["normal_train"]}, attack_count, f"cards|fold={fold}|normal")
    normal_items: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(task_result, CACHE, row, profile, center, scale, -1, normal_capacity[row["trace_name"]], 0) for row in roles["normal_train"]]
        for future in as_completed(futures):
            normal_items.append(future.result())
    if sum(len(item["samples"]) for item in normal_items) != attack_count:
        raise RuntimeError("normal 1:1 sampling contract failed without permitted fallback")
    x, y, group = canonical_arrays(selected_attack + normal_items)
    if int(y.sum()) != attack_count or int((1 - y).sum()) != attack_count:
        raise RuntimeError("assembled labels violate 1:1 normal-to-attack contract")
    return {"fold": fold, "x": x, "y": y, "group": group, "profile": profile, "center": center, "scale": scale, "roles": roles, "receipt": {"fold": fold, "sample_count": int(len(x)), "positive_count": int(y.sum()), "negative_count": int((1-y).sum()), "mechanism_count": len(mechanism_names), "input_sha256": {"x": sha256_bytes(np.ascontiguousarray(x).tobytes()), "y": sha256_bytes(np.ascontiguousarray(y).tobytes()), "group": sha256_bytes(np.ascontiguousarray(group).tobytes()), "center": sha256_bytes(np.ascontiguousarray(center).tobytes()), "scale": sha256_bytes(np.ascontiguousarray(scale).tobytes())}}}


def static_check(config_path: Path) -> dict[str, Any]:
    config = base.read_config(config_path)
    checks = {"cache_manifest_exists": (CACHE / "cache_manifest.json").is_file(), "workers_fixed": WORKERS == 6, "sequence_length_fixed": int(config["inputs"]["sequence_length_same_bus"]) == 16, "positive_caps_fixed": (int(config["sampling"]["positive_window_cap_per_mechanism"]), int(config["sampling"]["positive_trace_cap"])) == (10000, 4000), "normal_ratio_fixed": float(config["sampling"]["normal_to_attack_ratio"]) == 1.0, "no_outer_test_in_assembler_signature": True, "no_model_training": True}
    for fold in config["data"]["outer_folds"]:
        roles = roles_for_fold(config, fold)
        checks[f"fold_{fold}_roles_valid"] = all(len(roles[key]) > 0 for key in ("references", "normal_train", "attack_train"))
    return {"status": "TRAINING_ASSEMBLY_STATIC_PASS" if all(checks.values()) else "TRAINING_ASSEMBLY_STATIC_FAIL", "checks": checks, "scope": {"raw_archive_content_read": False, "outer_test_labels_read": False, "model_training": False}}


def completion_marker(status: str) -> str:
    """Map the two successful assembly states to the completion marker."""
    successful = {"TRAINING_ASSEMBLY_STATIC_PASS", "TRAINING_ASSEMBLY_PASS"}
    return "COMPLETED.json" if status in successful else "FAILED.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--assemble-training-side", action="store_true")
    args = parser.parse_args()
    if args.check == args.assemble_training_side:
        parser.error("pass exactly one of --check or --assemble-training-side")
    output, config_path = args.out.resolve(), args.config.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "script_snapshot.py"); shutil.copy2(config_path, output / "protocol_snapshot.json")
    try:
        config = base.read_config(config_path)
        if args.check:
            result = static_check(config_path)
        else:
            fold_receipts = []
            for fold in config["data"]["outer_folds"]:
                assembled = assemble_formal_fold(config, fold)
                fold_receipts.append(assembled["receipt"])
                del assembled
                gc.collect()
            result = {"status": "TRAINING_ASSEMBLY_PASS", "fold_receipts": fold_receipts,
                      "scope": {"roles_read": ["model_train_normal", "model_train_attack"], "normal_calibration_read": False, "outer_test_read": False, "uds_read": False, "ethernet_read": False, "model_training": False, "raw_archive_mutated": False}}
        dump(output / "result.json", result)
        marker = completion_marker(result["status"])
        dump(output / marker, {"status": result["status"]})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if marker == "COMPLETED.json" else 2
    except Exception as exc:
        dump(output / "FAILED.json", {"status": "FAILED", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
